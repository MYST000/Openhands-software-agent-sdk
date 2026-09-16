import base64
import json
import shlex
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .contracts import CommandResult
from .swe import export_patch_command, image_name
from .tracing import write_json

_COMMAND_RUNNER = r"""
import base64,ctypes,json,os,pathlib,signal,subprocess,sys,tempfile,time
p=json.loads(base64.b64decode(sys.argv[1]))
start=time.monotonic()
timed_out=False
ctypes.CDLL(None).prctl(36,1,0,0,0)
pidfile=p.get('pidfile')
if pidfile:pathlib.Path(pidfile).write_text(json.dumps({'pid':os.getpid(),'start':pathlib.Path('/proc/self/stat').read_text().split()[21]}))
def cancelled(signum,frame):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,cancelled)
def cleanup(proc):
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    # SIGKILL may need time to reclaim a large child address space.
    # Keep cleanup inside the controller's timeout + 15 second grace.
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        children=pathlib.Path('/proc/self/task/'+str(os.getpid())+'/children').read_text().split()
        if not children:break
        for child in children:
            try:os.kill(int(child),signal.SIGKILL)
            except ProcessLookupError:pass
        while True:
            try:
                if os.waitpid(-1,os.WNOHANG)[0]==0:break
            except ChildProcessError:break
        time.sleep(0.005)
    if pathlib.Path('/proc/self/task/'+str(os.getpid())+'/children').read_text().strip():
        raise RuntimeError('Could not quiesce command descendants')
with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
    proc=subprocess.Popen(['/bin/bash','--noprofile','--norc','-c',p['command']],cwd=p['cwd'],stdout=out,stderr=err,start_new_session=True)
    try:
        proc.wait(timeout=p['timeout'])
    except subprocess.TimeoutExpired:
        timed_out=True
        os.killpg(proc.pid,signal.SIGKILL)
        proc.wait()
    finally:
        cleanup(proc)
        if pidfile:pathlib.Path(pidfile).unlink(missing_ok=True)
    size_out=out.tell();size_err=err.tell()
    out.seek(0);err.seek(0)
    stdout=out.read(p['max_bytes']).decode('utf-8',errors='replace')
    stderr=err.read(p['max_bytes']).decode('utf-8',errors='replace')
print(json.dumps(dict(stdout=stdout,stderr=stderr,exit_code=proc.returncode,duration_ms=(time.monotonic()-start)*1000,timed_out=timed_out,truncated=size_out>p['max_bytes'] or size_err>p['max_bytes'],clock_domain='container-process')))
"""

_FILE_OPERATION = r"""
import base64,json,pathlib,sys
p=json.loads(base64.b64decode(sys.argv[1]))
root=pathlib.Path(p['root']).resolve()
a=p['action']
target=(root/a['path']).resolve()
try:target.relative_to(root)
except ValueError:raise ValueError('Path is outside the task repository')
if target==root or '.git' in target.relative_to(root).parts:raise ValueError('Cannot edit repository metadata')
cmd=a['command']
if cmd=='view':
    lines=target.read_text().splitlines()
    start=a.get('start_line',1);count=a.get('max_lines',200)
    if start<1 or count<1:raise ValueError('Invalid line range')
    print('\n'.join(str(i+1)+': '+lines[i] for i in range(start-1,min(len(lines),start-1+count))))
elif cmd=='create':
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as f:f.write(a['file_text'])
    print('Created '+str(target))
elif cmd=='str_replace':
    text=target.read_text();old=a['old_str'];new=a['new_str']
    if not old or text.count(old)!=1:raise ValueError('old_str must match exactly once')
    target.write_text(text.replace(old,new,1));print('Replaced one occurrence in '+str(target))
else:raise ValueError('Unknown file operation')
"""


def _encoded(payload):
    return base64.b64encode(json.dumps(payload).encode()).decode()


def command_argv(command, cwd, timeout, max_bytes, pidfile=None):
    return [
        "/usr/bin/python3",
        "-I",
        "-c",
        _COMMAND_RUNNER,
        _encoded(
            dict(command=command, cwd=cwd, timeout=timeout, max_bytes=max_bytes, pidfile=pidfile)
        ),
    ]


def file_operation_command(repo_dir, action):
    return shlex.join(
        [
            "/usr/bin/python3",
            "-I",
            "-c",
            _FILE_OPERATION,
            _encoded(dict(root=repo_dir, action=action)),
        ]
    )


class DockerEnvironment:
    """Owns one container; the actor has no host filesystem or Docker socket tool."""

    def __init__(self, config, task, artifact_dir, *, identity=None):
        self.config = config
        self.task = task
        self.artifact_dir = Path(artifact_dir)
        self.repo_dir = config.docker.repo_dir
        self.container = None
        self.client = None
        self.baseline_commit = task.public_metadata["base_commit"]
        self.initial_untracked = []
        self.active_pidfile = None
        self.active_exec_id = None
        self._prepare_started = False
        self._closed = False
        self._cleanup_complete = False
        identity = identity or {}
        self.metadata = dict(
            environment_id=uuid.uuid4().hex,
            task_id=task.task_id,
            run_id=identity.get("run_id", ""),
            attempt_id=identity.get("attempt_id", self.artifact_dir.parent.name),
            backend="docker",
            isolation="fresh-container-per-attempt",
        )

    def _lifecycle(self, status):
        self.metadata["lifecycle_status"] = status
        write_json(self.artifact_dir / "environment.json", self.metadata)

    def prepare(self):
        import docker

        if self._prepare_started or self._closed:
            raise RuntimeError(
                "DockerEnvironment is single-use; create a new environment for each attempt"
            )
        self._prepare_started = True
        self.client = docker.from_env(timeout=self.config.runtime.tool_timeout + 30)
        self.client.ping()
        image = image_name(self.task, self.config.docker.image_template)
        try:
            image_obj = self.client.images.get(image)
        except docker.errors.ImageNotFound:
            if not self.config.docker.pull_missing:
                raise RuntimeError(
                    f"Missing SWE image {image}. Use build-images --execute for local images "
                    "or prepare-images for registry images."
                ) from None
            image_obj = self.client.images.pull(image)
        lifetime = (
            self.config.runtime.task_timeout
            + self.config.llm.timeout * (self.config.llm.num_retries + 1)
            + 2 * self.config.runtime.tool_timeout
            + 300
        )
        self.metadata.update(
            image=image,
            image_id=image_obj.id,
            repo_digests=image_obj.attrs.get("RepoDigests", []),
            container_lifetime_s=lifetime,
        )
        self.container = self.client.containers.create(
            image_obj.id,
            command=["sleep", str(lifetime)],
            entrypoint=[],
            name="fp-swe-" + self.metadata["environment_id"],
            working_dir=self.repo_dir,
            network_disabled=self.config.docker.network_disabled,
            mem_limit=self.config.docker.memory_limit,
            nano_cpus=self.config.docker.nano_cpus,
            pids_limit=self.config.docker.pids_limit,
            init=True,
            auto_remove=not self.config.docker.keep_container,
            labels={
                "flowpilot.adapter": "swe-v1",
                "flowpilot.task_id": self.task.task_id,
                "flowpilot.run_id": self.metadata["run_id"],
                "flowpilot.attempt_id": self.metadata["attempt_id"],
                "flowpilot.environment_id": self.metadata["environment_id"],
            },
        )
        self.metadata["container_id"] = self.container.id
        self._lifecycle("created")
        self.container.start()
        base = self.task.public_metadata["base_commit"]
        self.checked("git merge-base --is-ancestor " + shlex.quote(base) + " HEAD")
        self.baseline_commit = self.checked("git rev-parse HEAD").strip()
        dirty = self.checked("git status --porcelain --untracked-files=no").strip()
        if dirty:
            raise RuntimeError("Image has tracked modifications before the actor starts")
        initial = self.checked("git ls-files --others --exclude-standard -z")
        self.initial_untracked = [p for p in initial.split("\0") if p]
        setup_patch = self.checked(
            f"git diff --binary --no-ext-diff {shlex.quote(base)} HEAD", max_bytes=20000000
        )
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "image_setup.patch").write_text(setup_patch)
        self.metadata.update(
            declared_base_commit=base,
            prepared_head=self.baseline_commit,
            initial_untracked=self.initial_untracked,
            patch_baseline="prepared_head",
            repo_dir=self.repo_dir,
        )
        self._lifecycle("ready")
        return self.metadata.copy()

    def execute(self, command, timeout=None, max_bytes=None, activate=True):
        if self._closed:
            raise RuntimeError("Task environment is closed")
        if self.container is None:
            raise RuntimeError("Container is not prepared")
        timeout = timeout or self.config.runtime.tool_timeout
        max_bytes = max_bytes or self.config.runtime.max_output_chars
        if activate and self.config.docker.conda_env:
            command = (
                "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate "
                + shlex.quote(self.config.docker.conda_env)
                + " && exec /bin/bash --noprofile --norc -c "
                + shlex.quote(command)
            )
        start = time.monotonic_ns()
        self.active_pidfile = "/tmp/fp-command-" + uuid.uuid4().hex + ".json"
        created = self.client.api.exec_create(
            self.container.id,
            command_argv(command, self.repo_dir, timeout, max_bytes, self.active_pidfile),
        )
        self.active_exec_id = created["Id"]
        output = self.client.api.exec_start(self.active_exec_id)
        code = self.client.api.exec_inspect(self.active_exec_id)["ExitCode"]
        if code != 0:
            raise RuntimeError(
                "Container executor failed: " + output.decode(errors="replace")[:1500]
            )
        result = CommandResult(**json.loads(output))
        self.active_pidfile = None
        self.active_exec_id = None
        self.last_rpc_duration_ms = (time.monotonic_ns() - start) / 1e6
        return result

    def quiesce(self):
        if self.container is None or self.active_pidfile is None:
            return
        if self.active_exec_id and not self.client.api.exec_inspect(self.active_exec_id)["Running"]:
            self.active_pidfile = self.active_exec_id = None
            return
        controller = """import json,os,pathlib,signal,sys,time
p=pathlib.Path(sys.argv[1])
for _ in range(200):
    if p.exists():break
    time.sleep(0.01)
else:raise RuntimeError('Running command has no supervisor marker')
if p.exists():
    data=json.loads(p.read_text());stat=pathlib.Path('/proc/'+str(data['pid'])+'/stat')
    if stat.exists() and stat.read_text().split()[21]==data['start']:
        os.kill(data['pid'],signal.SIGTERM)
        for _ in range(500):
            if not p.exists():break
            time.sleep(0.01)
        else:raise RuntimeError('Active command did not terminate; patch export forbidden')
    else:p.unlink()
"""
        reply = self.container.exec_run(
            ["/usr/bin/python3", "-I", "-c", controller, self.active_pidfile]
        )
        if reply.exit_code != 0:
            raise RuntimeError("Could not stop active container command before patch export")
        self.active_pidfile = None
        self.active_exec_id = None

    def checked(self, command, max_bytes=1000000):
        result = self.execute(command, max_bytes=max_bytes, activate=False)
        if result.exit_code or result.timed_out or result.truncated:
            raise RuntimeError(f"Workspace operation failed: {result.stderr[:1500]}")
        return result.stdout

    def export_patch(self):
        command = export_patch_command(self.baseline_commit)
        # Existing untracked image files are not actor changes. Git pathspecs are literal.
        if self.initial_untracked:
            excludes = " ".join(shlex.quote(":(top,literal)" + p) for p in self.initial_untracked)
            command = command.replace(
                "git add -A -- .\n",
                "git add -A -- .\n"
                + f"git rm --cached --ignore-unmatch -- {excludes} >/dev/null\n",
            )
        return self.checked(command, max_bytes=20000000)

    def close(self):
        import docker

        if self._cleanup_complete:
            return self.metadata.copy()
        self._closed = True
        try:
            status = "not_created"
            if self.container is not None:
                try:
                    if self.config.docker.keep_container:
                        self.container.stop(timeout=5)
                        status = "retained_stopped"
                    else:
                        self.container.remove(force=True)
                        status = "removed"
                except docker.errors.NotFound:
                    status = "already_removed"
            self._lifecycle(status)
            self._cleanup_complete = True
        except Exception:
            self._lifecycle("cleanup_failed")
            raise
        finally:
            if self.client is not None and self._cleanup_complete:
                self.client.close()
        return self.metadata.copy()


def command_dict(result):
    return asdict(result)
