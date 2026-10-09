"""Fixed-estimate, direct-engine benchmark transport; no agent loop or KV control."""

from dataclasses import dataclass
from typing import Any

from .config import JITServeConfig


@dataclass(frozen=True)
class JITServeContext:
    config: JITServeConfig
    conversation_id: str
    workflow_started_at: float

    @property
    def slo(self) -> dict[str, Any]:
        return {
            "backend": self.config.backend,
            "prediction_source": "caller_fixed",
            "deadline_origin": "task_start_before_environment_prepare",
            "workflow_started_at": self.workflow_started_at,
            "deadline": self.workflow_started_at + self.config.workflow_budget_seconds,
            "budget_seconds": self.config.workflow_budget_seconds,
            "job_id": f"job-{self.conversation_id}",
            "line_id": f"line-{self.conversation_id}",
            "conversation_id": self.conversation_id,
            "flowpilot_enabled": False,
        }

    def prepare(
        self,
        payload: dict[str, Any],
        *,
        logical_request_id: str,
        transport_id: str,
        attempt: int,
    ) -> dict[str, Any]:
        slo = self.slo
        xargs = {
            "jitserve_request_type": self.config.request_type,
            "jitserve_deadline": slo["deadline"],
            "jitserve_output_len": self.config.output_len,
            "jitserve_collection_id": slo["job_id"],
            "jitserve_ttft": self.config.ttft,
            "jitserve_tbt": self.config.tbt,
            "jitserve_client_request_id": transport_id,
        }
        payload["extra_headers"] = {
            **(payload.get("extra_headers") or {}),
            "X-Request-Id": transport_id,
        }
        if self.config.backend == "jitserve-v1-port":
            extra = dict(payload.get("extra_body") or {})
            extra["vllm_xargs"] = {**extra.get("vllm_xargs", {}), **xargs}
            payload["extra_body"] = extra
        return {
            **slo,
            "logical_request_id": logical_request_id,
            "request_id": transport_id,
            "transport_attempt": attempt,
            "slo_parameters": xargs,
            "slo_sent_to_engine": self.config.backend == "jitserve-v1-port",
        }
