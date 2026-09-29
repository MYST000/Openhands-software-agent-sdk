from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    dataset_id: str
    revision: str
    split: str
    task_id: str
    instruction: str
    public_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    timed_out: bool = False
    truncated: bool = False
    clock_domain: str = "executor"


class ConfigurationError(ValueError):
    pass
