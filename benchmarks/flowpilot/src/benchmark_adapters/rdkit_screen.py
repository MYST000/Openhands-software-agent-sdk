"""Native screen_target tool and runner for the official RDKit screening workload."""

import argparse
import hashlib
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

from openhands.sdk import Action, Agent, Conversation, Observation, ToolDefinition
from openhands.sdk.tool import Tool, ToolAnnotations, ToolExecutor, register_tool
from pydantic import Field, SecretStr

from .sdk_bridge import Binding, RecordedLLM, bind, get_binding, unbind
from .tracing import Budget, TraceRecorder, write_json


class ScreenAction(Action):
    target: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    fingerprint: Literal["rdk7", "ecfp4"]
    num_queries: int = 5
    num_splits: int = 50


class ScreenObservation(Observation):
    pass


class ScreenEnvironment:
    """One persistent, serial worker per task; no fingerprint or result cache."""

    def __init__(self, source_root, directory, *, dataset, target, fingerprint, python):
        self.source_root = Path(source_root).resolve()
        self.directory = Path(directory).resolve()
        self.dataset = dataset
        self.case_id = f"platform-{dataset}{target}-{fingerprint}"
        self.arguments = dict(target=target, fingerprint=fingerprint, num_queries=5, num_splits=50)
        self.python = python
        self.process = None
        self.log = None
        self.outputs = []

    def _receive(self, timeout):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            self.close()
            raise TimeoutError("RDKit worker exceeded the deadline")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"RDKit worker exited; see {self.directory / 'worker.log'}")
        return json.loads(line)

    def prepare(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = (self.directory / "worker.log").open("w")
        try:
            self.process = subprocess.Popen(
                [
                    self.python,
                    "-u",
                    str(Path(__file__).with_name("rdkit_screen_worker.py")),
                    "--source-root",
                    str(self.source_root),
                    "--dataset",
                    self.dataset,
                    "--target",
                    self.arguments["target"],
                    "--fingerprint",
                    self.arguments["fingerprint"],
                    "--output",
                    str(self.directory / "artifacts"),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )
            ready = self._receive(90)
            if ready.get("ready") is not True:
                raise RuntimeError(f"RDKit worker startup failed: {ready}")
            write_json(self.directory / "worker_ready.json", ready)
            return {"backend": "rdkit-official-screen-worker", **ready}
        except Exception:
            self.close()
            raise

    def screen(self, arguments, timeout):
        if arguments != self.arguments:
            raise ValueError(f"Use exactly the requested parameters: {self.arguments}")
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("RDKit worker is not running")
        self.process.stdin.write(json.dumps({"case_id": self.case_id}) + "\n")
        self.process.stdin.flush()
        result = self._receive(timeout)
        self.outputs.append(result)
        write_json(self.directory / "worker_outputs.json", self.outputs)
        if result.get("ok") is not True:
            raise RuntimeError(f"RDKit screening failed: {result.get('error')}")
        payload = result["payload"]
        artifact = Path(payload["artifact_scores"])
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != payload["artifact_sha256"]:
            raise ValueError("RDKit score artifact checksum mismatch")
        return result

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                stream.close()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None


class ScreenExecutor(ToolExecutor):
    def __init__(self, binding):
        self.binding = binding

    def __call__(self, action, conversation=None):
        b = self.binding
        arguments = action.model_dump(mode="json", exclude={"kind", "summary", "security_risk"})
        b.budget.tool()
        link = b.recorder.start_tool("screen_target", arguments)
        started = time.monotonic()
        try:
            with b.recorder.activity("tool"):
                result = b.environment.screen(arguments, min(b.tool_timeout, b.budget.remaining()))
            text = json.dumps(result["payload"])
            elapsed = (time.monotonic() - started) * 1000
            b.recorder.emit(
                "tool_end",
                tool_name="screen_target",
                **link,
                effective_arguments=arguments,
                round_trip_ms=elapsed,
                executor_duration_ms=elapsed,
                worker_stages=result["stages"],
                model_observation=text,
                outcome={"exit_code": 0},
                timed_out=False,
            )
            return ScreenObservation.from_text(text)
        except Exception as exc:
            b.recorder.emit(
                "tool_error",
                tool_name="screen_target",
                **link,
                effective_arguments=arguments,
                error_type=type(exc).__name__,
                error=str(exc),
                round_trip_ms=(time.monotonic() - started) * 1000,
                timed_out=isinstance(exc, TimeoutError),
            )
            return ScreenObservation.from_text(str(exc), is_error=True)


class ScreenTargetTool(ToolDefinition):
    name = "screen_target"

    @classmethod
    def create(cls, conv_state=None, **params):
        return [
            cls(
                action_type=ScreenAction,
                observation_type=ScreenObservation,
                description="Screen the task's official RDKit target using Dice similarity and max "
                "fusion, five active queries and all fifty published data_sets_I splits. Return "
                "first-split top five IDs, candidates per split and candidate occurrences over all splits.",
                executor=ScreenExecutor(get_binding(params["binding_key"])),
                annotations=ToolAnnotations(readOnlyHint=True),
            )
        ]


register_tool(ScreenTargetTool.name, ScreenTargetTool)


def screening_tools(binding_key):
    return [Tool(name=ScreenTargetTool.name, params={"binding_key": binding_key})]


def run_screen(
    directory,
    *,
    source_root,
    dataset,
    target,
    fingerprint,
    python,
    model,
    base_url,
    api_key,
    check_tool=False,
):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    env = ScreenEnvironment(
        source_root,
        directory,
        dataset=dataset,
        target=target,
        fingerprint=fingerprint,
        python=python,
    )
    recorder = TraceRecorder(directory, {"task_id": env.case_id, "benchmark": "rdkit-screen"})
    recorder.environment_tool_names = {"screen_target"}
    budget = Budget(8, 10, 1800)
    binding = Binding(env, recorder, budget, 600, 16000)
    key = bind(binding)
    conversation = None
    started = time.monotonic()
    instruction = (
        f"Run virtual screening on official {dataset} target {target}, fingerprint {fingerprint}, "
        "Dice similarity, 5 active queries, all 50 published data_sets_I splits and max fusion. "
        "Do not change the parameters. Report first_split_top5 IDs, test_candidates_per_split "
        "and test_candidate_occurrences_all_splits separately. The latter counts repeated "
        "appearances across splits, not unique molecules. Call finish after obtaining results."
    )
    write_json(
        directory / "public_task.json",
        {
            "instruction": instruction,
            "dataset": dataset,
            **env.arguments,
            "protocol": "rdkit-screen-counts-v2",
            "cache_enabled": False,
        },
    )
    recorder.emit("task_start")
    try:
        recorder.emit("environment_ready", metadata=env.prepare())
        if check_tool:
            tool = ScreenTargetTool.create(binding_key=key)[0]
            observation = tool.executor(ScreenAction(**env.arguments))
            if observation.is_error:
                raise RuntimeError(observation.text)
            result = {"execution_status": "tool_checked", "payload": json.loads(observation.text)}
        else:
            llm = RecordedLLM(
                model=model,
                base_url=base_url,
                api_key=SecretStr(api_key),
                usage_id="actor",
                temperature=1.0,
                top_p=0.95,
                seed=41,
                max_input_tokens=258048,
                max_output_tokens=4096,
                timeout=900,
                num_retries=0,
                native_tool_calling=True,
                stream=False,
                caching_prompt=False,
                litellm_extra_body={
                    "top_k": 20,
                    "min_p": 0,
                    "presence_penalty": 1.5,
                    "repetition_penalty": 1,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            llm.attach(recorder, budget)
            agent = Agent(
                llm=llm,
                tools=screening_tools(key),
                include_default_tools=["FinishTool"],
                condenser=None,
                tool_concurrency_limit=1,
                system_prompt="Use the supplied screening tool. Do not invent numbers. "
                "Report the requested IDs and both counts as concise JSON, then call finish.",
            )
            conversation = Conversation(
                agent=agent,
                workspace=str(directory / "workspace"),
                callbacks=[recorder.sdk_event],
                max_iteration_per_run=10,
                persistence_dir=str(directory / "sdk_state"),
                visualizer=None,
            )
            conversation.send_message(instruction)
            conversation.run()
            status = conversation.state.execution_status.value
            result = {
                "execution_status": "completed" if status == "finished" else status,
                "final_text": recorder.final_text,
            }
        result.update(
            jct_s=time.monotonic() - started, tool_calls=recorder.executed_counts["screen_target"]
        )
        write_json(directory / "result.json", result)
        return result
    finally:
        try:
            if conversation is not None:
                conversation.close()
        finally:
            env.close()
            recorder.close()
            unbind(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("MUV", "DUD", "ChEMBL"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--fingerprint", choices=("rdk7", "ecfp4"), default="rdk7")
    parser.add_argument("--rdkit-python", default=sys.executable)
    parser.add_argument("--model", default="openai/Qwen3.5-27B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-tool", action="store_true", help="Execute the tool without an LLM")
    args = parser.parse_args()
    print(
        json.dumps(
            run_screen(
                args.output,
                source_root=args.source_root,
                dataset=args.dataset,
                target=args.target,
                fingerprint=args.fingerprint,
                python=args.rdkit_python,
                model=args.model,
                base_url=args.base_url,
                api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
                check_tool=args.check_tool,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
