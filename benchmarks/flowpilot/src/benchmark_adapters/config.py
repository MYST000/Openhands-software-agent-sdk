import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .contracts import ConfigurationError

SDK_COMMIT = "c4f3ea625f8fbe72f2af12e879651dcff4ba4ef9"
SDK_PATH = str(Path(__file__).resolve().parents[4])


@dataclass(frozen=True)
class DatasetConfig:
    kind: str = "swe"
    id: str = "princeton-nlp/SWE-bench"
    path: str = ""
    revision: str = ""
    split: str = "dev"
    sha256: str = ""
    setting: str = "openhands-code-v1"


@dataclass(frozen=True)
class LLMConfig:
    model: str = "openai/qwen3.5-9b"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    enable_thinking: bool | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int = 4096
    timeout: int = 180
    num_retries: int = 2
    native_tool_calling: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    sdk_path: str = SDK_PATH
    sdk_commit: str = SDK_COMMIT
    max_iterations: int = 60
    max_tool_calls: int = 120
    max_llm_requests: int = 80
    task_timeout: int = 3600
    tool_timeout: int = 120
    max_output_chars: int = 16000
    runs_dir: str = "/root/predictor_exp/runs/benchmark_adapters"


@dataclass(frozen=True)
class DockerConfig:
    image_template: str = "swebench/sweb.eval.x86_64.{image_id}:latest"
    repo_dir: str = "/testbed"
    conda_env: str = "testbed"
    network_disabled: bool = True
    memory_limit: str = "8g"
    nano_cpus: int = 2000000000
    pids_limit: int = 512
    pull_missing: bool = False
    keep_container: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    backend: str = "sqlite"
    mcp_url: str = "http://127.0.0.1:8123/mcp"
    index_path: str = ""
    corpus_revision: str = ""
    server_policy_revision: str = ""
    top_k: int = 5
    snippet_chars: int = 1200
    read_chars: int = 6000


@dataclass(frozen=True)
class EvaluationConfig:
    python: str = ""
    harness_path: str = "/root/predictor_exp/repos/SWE-bench"
    workers: int = 1
    timeout: int = 1800
    namespace: str = "swebench"
    image_tag: str = "latest"
    hotpot_script: str = ""
    browsecomp_repo: str = "/root/predictor_exp/repos/BrowseComp-Plus"
    judge_model: str = "Qwen/Qwen3-32B"


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text())
    classes = dict(
        dataset=DatasetConfig,
        llm=LLMConfig,
        runtime=RuntimeConfig,
        docker=DockerConfig,
        retrieval=RetrievalConfig,
        evaluation=EvaluationConfig,
    )
    if unknown := raw.keys() - classes.keys():
        raise ConfigurationError(f"Unknown config sections: {sorted(unknown)}")
    objects = {}
    for section, cls in classes.items():
        values = raw.get(section, {})
        if not isinstance(values, dict):
            raise ConfigurationError(f"{section} must be a TOML table")
        if unknown := values.keys() - {f.name for f in fields(cls)}:
            raise ConfigurationError(f"Unknown {section} keys: {sorted(unknown)}")
        defaults = cls()
        for key, value in values.items():
            expected = type(getattr(defaults, key))
            if section == "llm" and getattr(defaults, key) is None:
                expected = {
                    "max_input_tokens": int,
                    "top_p": float,
                    "top_k": int,
                    "seed": int,
                    "presence_penalty": float,
                    "min_p": float,
                    "repetition_penalty": float,
                    "enable_thinking": bool,
                }[key]
            if expected is float and type(value) in (int, float):
                continue
            if type(value) is not expected:
                raise ConfigurationError(f"{section}.{key} must be {expected.__name__}")
        host_paths = {
            "dataset": {"path"},
            "runtime": {"sdk_path", "runs_dir"},
            "retrieval": {"index_path"},
            "evaluation": {"python", "harness_path", "hotpot_script", "browsecomp_repo"},
        }
        for key in list(values):
            if values[key] and key in host_paths.get(section, set()):
                host_path = path.parent / values[key]
                # Resolving a venv Python symlink bypasses pyvenv.cfg and loses its packages.
                values[key] = str(host_path.absolute() if key == "python" else host_path.resolve())
        objects[section] = cls(**values)
    cfg = Config(**objects)
    if cfg.dataset.kind not in {
        "swe",
        "hotpot",
        "browsecomp",
        "quixbugs",
        "livecodebench",
        "classeval",
    }:
        raise ConfigurationError("Unsupported dataset.kind")
    if cfg.dataset.id.endswith("SWE-bench_Verified") and cfg.dataset.split != "test":
        raise ConfigurationError("SWE-bench Verified only has the test split")
    supported = {
        "swe": {"princeton-nlp/SWE-bench", "princeton-nlp/SWE-bench_Verified"},
        "hotpot": {"hotpotqa"},
        "browsecomp": {"Tevatron/browsecomp-plus"},
        "quixbugs": {"jkoppel/QuixBugs"},
        "livecodebench": {"livecodebench/code_generation_lite"},
        "classeval": {"FudanSELab/ClassEval"},
    }
    if cfg.dataset.id not in supported[cfg.dataset.kind]:
        raise ConfigurationError("Unsupported dataset identity for selected adapter")
    settings = {
        "swe": "openhands-code-v1",
        "hotpot": "fullwiki-fixed-corpus-v1",
        "browsecomp": "openhands-search-read-v1",
        "quixbugs": "openhands-python-code-v1",
        "livecodebench": "openhands-python-code-v1",
        "classeval": "openhands-python-code-v1",
    }
    if cfg.dataset.setting != settings[cfg.dataset.kind]:
        raise ConfigurationError(
            f"Unsupported dataset.setting; expected {settings[cfg.dataset.kind]}"
        )
    if cfg.dataset.split not in {"train", "dev", "test"}:
        raise ConfigurationError("Unknown dataset split")
    if cfg.retrieval.backend not in {"sqlite", "browsecomp_mcp", "hotpot_rpc"}:
        raise ConfigurationError("Unknown retrieval backend")
    if cfg.retrieval.backend == "browsecomp_mcp":
        if cfg.dataset.kind != "browsecomp":
            raise ConfigurationError("The official BrowseComp MCP backend only supports BrowseComp")
        if not cfg.retrieval.mcp_url.startswith(("http://", "https://")):
            raise ConfigurationError("retrieval.mcp_url must be an HTTP(S) MCP endpoint")
    if cfg.retrieval.backend == "hotpot_rpc":
        if cfg.dataset.kind != "hotpot":
            raise ConfigurationError("The Hotpot RPC backend only supports Hotpot")
        if not cfg.retrieval.mcp_url.startswith(("http://", "https://")):
            raise ConfigurationError("retrieval.mcp_url must be an HTTP(S) RPC endpoint")
    if cfg.dataset.kind == "browsecomp" and cfg.dataset.split != "test":
        raise ConfigurationError("BrowseComp-Plus question data only has test split")
    if not cfg.dataset.path or not cfg.dataset.revision:
        raise ConfigurationError("dataset.path and dataset.revision are required")
    for section in (cfg.runtime, cfg.retrieval, cfg.evaluation, cfg.docker):
        for f in fields(section):
            value = getattr(section, f.name)
            if type(value) is int and value <= 0:
                raise ConfigurationError(f"{f.name} must be positive")
    if cfg.llm.max_output_tokens <= 0 or cfg.llm.timeout <= 0 or cfg.llm.num_retries < 0:
        raise ConfigurationError("Invalid LLM token/timeout/retry budget")
    if cfg.llm.max_input_tokens is not None and cfg.llm.max_input_tokens <= 0:
        raise ConfigurationError("max_input_tokens must be positive")
    if cfg.llm.temperature < 0:
        raise ConfigurationError("temperature must be nonnegative")
    for name in ("top_p", "min_p"):
        value = getattr(cfg.llm, name)
        if value is not None and not 0 <= value <= 1:
            raise ConfigurationError(f"{name} must be in 0..1")
    if cfg.llm.top_k is not None and cfg.llm.top_k < -1:
        raise ConfigurationError("top_k must be at least -1")
    if cfg.llm.repetition_penalty is not None and cfg.llm.repetition_penalty <= 0:
        raise ConfigurationError("repetition_penalty must be positive")
    if cfg.llm.presence_penalty is not None and not -2 <= cfg.llm.presence_penalty <= 2:
        raise ConfigurationError("presence_penalty must be in -2..2")
    if not cfg.llm.native_tool_calling:
        raise ConfigurationError("Prediction tracing requires llm.native_tool_calling=true")
    if cfg.runtime.sdk_commit != SDK_COMMIT:
        raise ConfigurationError(f"This implementation supports the pinned SDK {SDK_COMMIT}")
    if not cfg.docker.repo_dir.startswith("/") or cfg.docker.repo_dir == "/":
        raise ConfigurationError("docker.repo_dir must be an absolute task directory")
    return cfg
