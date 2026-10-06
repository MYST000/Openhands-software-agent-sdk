"""Default-off FlowPilot adapter for correlation, telemetry, and safe reuse."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from pydantic import SecretStr

from openhands.sdk.event import ObservationEvent
from openhands.sdk.flowpilot_reuse import (
    canonical_input,
    canonical_json,
    payload_digest,
)
from openhands.sdk.llm import TextContent
from openhands.sdk.logger import get_logger
from openhands.sdk.mcp.definition import MCPToolAction
from openhands.sdk.tool.schema import Observation
from openhands.sdk.utils.cipher import Cipher


if TYPE_CHECKING:
    from openhands.sdk.event import ActionEvent, Event
    from openhands.sdk.io import FileStore

logger = get_logger(__name__)


class FlowPilotReusedObservation(Observation):
    """Provider-visible Tool result reconstructed from a validated reuse hit."""


@dataclass(frozen=True, slots=True)
class FlowPilotConfig:
    """Conversation-scoped FlowPilot adapter configuration."""

    enabled: bool = False
    gateway_url: str = ""
    api_key: str = ""
    job_id: str = ""
    line_id: str = ""
    deployment_id: str | None = None
    namespace_id: str | None = None
    root_conversation_id: str | None = None
    parent_line_id: str | None = None
    spawn_id: str | None = None
    parent_conversation_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    parent_action_id: str | None = None
    timeout: float = 2.0
    include_auxiliary_llms: bool = False
    exact_reuse_enabled: bool = False
    semantic_reuse_enabled: bool = False
    reusable_web_tools: tuple[str, ...] = ()
    locale: str = "und"
    language: str = "und"
    region: str = "global"
    safe_search_policy: str = "default"
    time_sensitivity_class: str = "standard"
    data_source_constraints: tuple[str, ...] = ()
    reuse_output_budget_bytes: int | None = None
    reuse_wait_timeout: float = 30.0
    reuse_poll_interval: float = 0.05
    deferred_context_enabled: bool = False
    delegation_lease_seconds: float = 30.0
    deferred_max_messages: int = 32
    deferred_max_bytes: int = 1_000_000
    max_internal_continuations: int = 8
    deferred_delta_ttl_seconds: float = 300.0

    @property
    def control_base_url(self) -> str:
        value = self.gateway_url.rstrip("/")
        return value[:-3] if value.endswith("/v1") else value

    @property
    def llm_base_url(self) -> str:
        return f"{self.control_base_url}/v1"

    def child(
        self, *, line_id: str, parent_conversation_id: str, spawn_id: str
    ) -> FlowPilotConfig:
        return replace(
            self,
            line_id=line_id,
            parent_conversation_id=parent_conversation_id,
            parent_line_id=self.line_id,
            spawn_id=spawn_id,
        )

    @property
    def reuse_protocol_version(self) -> str:
        return (
            "flowpilot-phase3-reuse-v3"
            if self.semantic_reuse_enabled
            else "flowpilot-phase1-reuse-v3"
        )

    def validate(self, *, tool_concurrency_limit: int) -> None:
        if not self.enabled:
            return
        if tool_concurrency_limit != 1:
            raise ValueError(
                "FlowPilot Phase 0 requires tool_concurrency_limit == 1; "
                f"got {tool_concurrency_limit}"
            )
        missing = [
            name
            for name in ("gateway_url", "api_key", "job_id", "line_id")
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(
                f"FlowPilot configuration is missing: {', '.join(missing)}"
            )
        if self.parent_line_id is not None and not self.parent_conversation_id:
            raise ValueError("parent_line_id requires parent_conversation_id")
        if self.timeout <= 0:
            raise ValueError("FlowPilot timeout must be positive")
        if self.exact_reuse_enabled and not self.reusable_web_tools:
            raise ValueError(
                "FlowPilot exact reuse requires an explicit reusable_web_tools registry"
            )
        if self.semantic_reuse_enabled and not self.exact_reuse_enabled:
            raise ValueError(
                "FlowPilot semantic reuse requires exact reuse to be enabled"
            )
        if self.deferred_context_enabled and not self.exact_reuse_enabled:
            raise ValueError(
                "FlowPilot deferred context requires exact reuse to be enabled"
            )
        if self.reuse_wait_timeout <= 0 or self.reuse_poll_interval <= 0:
            raise ValueError("FlowPilot reuse wait and poll intervals must be positive")
        if (
            self.reuse_output_budget_bytes is not None
            and self.reuse_output_budget_bytes <= 0
        ):
            raise ValueError("FlowPilot reuse output budget must be positive")
        if self.delegation_lease_seconds <= 0:
            raise ValueError("FlowPilot delegation lease must be positive")
        if self.deferred_max_messages <= 0 or self.deferred_max_bytes <= 0:
            raise ValueError("FlowPilot deferred context limits must be positive")
        if self.max_internal_continuations <= 0:
            raise ValueError("FlowPilot continuation limit must be positive")
        if self.deferred_delta_ttl_seconds <= 0:
            raise ValueError("FlowPilot deferred delta TTL must be positive")


@dataclass(frozen=True, slots=True)
class FlowPilotRequestIdentity:
    job_id: str
    line_id: str
    tail_request_id: str
    request_id: str
    llm_call_id: str
    attempt: int
    tail_version: int
    context_epoch: int
    context_sequence: int
    base_context_cursor: str
    context_digest: str
    conversation_id: str | None = None
    parent_conversation_id: str | None = None
    parent_line_id: str | None = None
    spawn_id: str | None = None
    deployment_id: str | None = None
    namespace_id: str | None = None
    origin: Literal["agent", "scheduler_delegated"] = "agent"
    delegation_lease_id: str | None = None

    def headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "x-flowpilot-api-key": api_key,
            "x-flowpilot-protocol-version": "flowpilot-phase0-v2",
            "x-flowpilot-job-id": self.job_id,
            "x-flowpilot-line-id": self.line_id,
            "x-flowpilot-tail-request-id": self.tail_request_id,
            "x-flowpilot-request-id": self.request_id,
            "x-flowpilot-request-attempt": str(self.attempt),
            "x-flowpilot-llm-call-id": self.llm_call_id,
            "x-flowpilot-tail-version": str(self.tail_version),
            "x-flowpilot-context-epoch": str(self.context_epoch),
            "x-flowpilot-context-sequence": str(self.context_sequence),
            "x-flowpilot-context-cursor": self.base_context_cursor,
            "x-flowpilot-context-digest": self.context_digest,
            "x-flowpilot-request-origin": self.origin,
        }
        if self.deployment_id is not None:
            headers["x-flowpilot-deployment-id"] = self.deployment_id
        if self.namespace_id is not None:
            headers["x-flowpilot-namespace-id"] = self.namespace_id
        for name, value in (
            ("x-flowpilot-conversation-id", self.conversation_id),
            ("x-flowpilot-parent-conversation-id", self.parent_conversation_id),
            ("x-flowpilot-parent-line-id", self.parent_line_id),
            ("x-flowpilot-spawn-id", self.spawn_id),
        ):
            if value is not None:
                headers[name] = value
        if self.delegation_lease_id is not None:
            headers["x-flowpilot-delegation-lease-id"] = self.delegation_lease_id
        return headers


@dataclass(frozen=True, slots=True)
class FlowPilotDCSReference:
    job_id: str
    line_id: str
    context_epoch: int
    lease_id: str
    base_context_cursor: str
    delta_digest: str

    def payload(self) -> dict[str, Any]:
        return {
            "protocol_version": "flowpilot-phase2-dcs-v2",
            **asdict(self),
        }


@dataclass(slots=True)
class FlowPilotRuntime:
    """Process-local identity, telemetry, and conservative-reuse client."""

    config: FlowPilotConfig
    conversation_id: str
    tail_version: int = 0
    context_epoch: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _active_identity: FlowPilotRequestIdentity | None = None
    _transport_attempts: int = 0
    _tool_identity: FlowPilotRequestIdentity | None = None
    _attempts: dict[str, int] = field(default_factory=dict)
    _active_tools: dict[str, tuple[FlowPilotRequestIdentity, int, float]] = field(
        default_factory=dict
    )
    _leader_bindings: dict[str, tuple[FlowPilotRequestIdentity, str]] = field(
        default_factory=dict
    )
    _reuse_cancellations: dict[str, threading.Event] = field(default_factory=dict)
    _waiting_reuses: dict[str, tuple[str, dict[str, str]]] = field(default_factory=dict)
    _dcs_policy_version: int = 0
    _dcs_reference: FlowPilotDCSReference | None = None
    _dcs_base_context_sequence: int = 0
    _dcs_base_context_digest: str | None = None
    _dcs_last_seq: int = 0
    _dcs_required_sync_reason: str | None = None
    _gateway_reuse_policy: dict[str, Any] | None = None
    _gateway_decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    _gateway_batches: list[dict[str, Any]] = field(default_factory=list)
    _gateway_response_deferred: bool = False
    _leader_credentials: dict[str, dict[str, Any]] = field(default_factory=dict)
    _pending_publications: dict[str, dict[str, Any]] = field(default_factory=dict)
    _file_store: FileStore | None = None
    _sync_response: dict[str, Any] | None = None
    _sync_reference: FlowPilotDCSReference | None = None
    _dependency_version: int = 0
    _waiting_on_lines: set[str] = field(default_factory=set)
    _registered: bool = False

    _RECOVERY_DIR = "flowpilot/recovery"

    def attach_file_store(self, file_store: FileStore) -> None:
        """Attach the conversation store used for restart-safe DCS recovery."""
        self._file_store = file_store

    def _recovery_manifest_path(self) -> str:
        return f"{self._RECOVERY_DIR}/pending.json"

    def _recovery_payload_path(self, batch_id: str) -> str:
        return f"{self._RECOVERY_DIR}/{batch_id}.payload.json"

    def _recovery_cipher(self) -> Cipher:
        material = ":".join(
            (
                self.config.api_key,
                self.config.job_id,
                self.config.line_id,
                self.conversation_id,
            )
        )
        return Cipher(material)

    def _read_recovery_manifest(self) -> dict[str, Any] | None:
        if self._file_store is None or not self._file_store.exists(
            self._recovery_manifest_path()
        ):
            return None
        try:
            value = json.loads(self._file_store.read(self._recovery_manifest_path()))
        except (OSError, ValueError, TypeError):
            logger.warning(
                "Ignoring malformed FlowPilot recovery manifest", exc_info=True
            )
            return None
        return value if isinstance(value, dict) else None

    def _clear_recovery_manifest(self, manifest: dict[str, Any] | None = None) -> None:
        if self._file_store is None:
            return
        try:
            self._file_store.delete(self._recovery_manifest_path())
            if manifest is not None and isinstance(manifest.get("payload_file"), str):
                self._file_store.delete(manifest["payload_file"])
        except Exception:
            logger.warning("FlowPilot recovery cleanup failed", exc_info=True)

    def _record_recovery_applied_context(
        self, manifest: dict[str, Any] | None, cursor: str, digest: str
    ) -> None:
        """Record the local commit before its DCS ACK can advance the WAL."""
        if self._file_store is None or manifest is None:
            return
        manifest["applied_context_cursor"] = cursor
        manifest["applied_context_digest"] = digest
        self._file_store.write(
            self._recovery_manifest_path(), json.dumps(manifest, separators=(",", ":"))
        )

    def persist_recovery_record(
        self,
        *,
        messages: tuple[dict[str, Any], ...],
        events: list[Event],
        pending_batches: tuple[tuple[list[Event], tuple[dict[str, Any], ...]], ...]
        | None = None,
        first_chunk_batch_count: int = 1,
    ) -> None:
        """Persist a metadata-only manifest before applying hidden events.

        Full provider/event payloads live in the normal OpenHands event store
        payload file; the manifest contains only stable IDs and digests, so it
        cannot become a prompt or Tool-result log.
        """
        if self._file_store is None:
            raise RuntimeError("FlowPilot DCS recovery requires a persistent FileStore")
        response = self._sync_response
        reference = self._sync_reference or self._require_dcs_reference()
        if response is None:
            raise RuntimeError("FlowPilot sync metadata is not available")
        batch_id = str(uuid.uuid4())
        recovery_batches = pending_batches or ((events, messages),)
        if not recovery_batches or not 0 < first_chunk_batch_count <= len(
            recovery_batches
        ):
            raise ValueError("FlowPilot recovery batch metadata is malformed")

        serialized_batches: list[dict[str, Any]] = []
        for batch_events, batch_messages in recovery_batches:
            provider_payload = [
                _redact_provider_message(message) for message in batch_messages
            ]
            event_payloads = [
                event.model_dump(mode="json", exclude_none=True)
                for event in batch_events
            ]
            encrypted_events = self._recovery_cipher().encrypt(
                SecretStr(json.dumps(event_payloads, separators=(",", ":")))
            )
            if encrypted_events is None:
                raise RuntimeError("FlowPilot could not encrypt recovery events")
            serialized_batches.append(
                {
                    "encrypted_events": encrypted_events,
                    "messages": provider_payload,
                    "event_ids": [event.id for event in batch_events],
                    "event_digests": [
                        _stable_digest(event.model_dump(mode="json", exclude_none=True))
                        for event in batch_events
                    ],
                }
            )

        provider_payload = serialized_batches[0]["messages"]
        payload_file = self._recovery_payload_path(batch_id)
        payload = {
            # Keep the first-batch fields for compatibility with v1 manifests.
            "encrypted_events": serialized_batches[0]["encrypted_events"],
            "messages": provider_payload,
            "batches": serialized_batches,
        }
        first_batch = serialized_batches[0]
        manifest = {
            "protocol_version": "flowpilot-phase2-recovery-v2",
            "batch_id": batch_id,
            "event_ids": first_batch["event_ids"],
            "event_digests": first_batch["event_digests"],
            "provider_message_count": len(messages),
            "provider_batch_digest": _stable_digest(provider_payload),
            "batch_count": len(serialized_batches),
            "first_chunk_batch_count": first_chunk_batch_count,
            "payload_file": payload_file,
            "dcs_reference": reference.payload(),
            "policy_version": self._dcs_policy_version,
            "base_context_sequence": self._dcs_base_context_sequence,
            "base_context_digest": self._dcs_base_context_digest,
            "first_seq": response.get("first_seq"),
            "last_seq": response.get("last_seq"),
            "delta_digest": response.get("delta_digest"),
            "base_head_id": reference.base_context_cursor,
            "last_event_id": events[-1].id if events else reference.base_context_cursor,
        }
        self._file_store.write(payload_file, json.dumps(payload, separators=(",", ":")))
        self._file_store.write(
            self._recovery_manifest_path(), json.dumps(manifest, separators=(",", ":"))
        )

    def _load_recovery_batches(
        self, manifest: dict[str, Any]
    ) -> list[tuple[list[Event], tuple[dict[str, Any], ...]]]:
        if self._file_store is None:
            raise RuntimeError("FlowPilot DCS recovery requires a persistent FileStore")
        payload_file = manifest.get("payload_file")
        if not isinstance(payload_file, str):
            raise ValueError("FlowPilot recovery payload reference is malformed")
        raw = json.loads(self._file_store.read(payload_file))
        raw_batches = raw.get("batches")
        if not isinstance(raw_batches, list):
            raw_batches = [
                {
                    "encrypted_events": raw.get("encrypted_events"),
                    "messages": raw.get("messages"),
                    "event_ids": manifest.get("event_ids"),
                    "event_digests": manifest.get("event_digests"),
                }
            ]
        from openhands.sdk.event import Event

        batches: list[tuple[list[Event], tuple[dict[str, Any], ...]]] = []
        for raw_batch in raw_batches:
            if not isinstance(raw_batch, dict):
                raise ValueError("FlowPilot recovery batches are malformed")
            encrypted_events = raw_batch.get("encrypted_events")
            raw_messages = raw_batch.get("messages")
            if not isinstance(encrypted_events, str) or not isinstance(
                raw_messages, list
            ):
                raise ValueError("FlowPilot recovery batch is malformed")
            decrypted = self._recovery_cipher().decrypt(encrypted_events)
            if decrypted is None:
                raise ValueError("FlowPilot recovery event payload cannot be decrypted")
            raw_events = json.loads(decrypted.get_secret_value())
            if not isinstance(raw_events, list):
                raise ValueError("FlowPilot recovery events are malformed")
            events = [Event.model_validate(item) for item in raw_events]
            expected_ids = raw_batch.get("event_ids")
            expected_digests = raw_batch.get("event_digests")
            if not isinstance(expected_ids, list) or not isinstance(
                expected_digests, list
            ):
                raise ValueError("FlowPilot recovery event metadata is malformed")
            if [event.id for event in events] != expected_ids:
                raise ValueError("FlowPilot recovery event IDs do not match manifest")
            if [
                _stable_digest(event.model_dump(mode="json", exclude_none=True))
                for event in events
            ] != expected_digests:
                raise ValueError(
                    "FlowPilot recovery event contents do not match manifest"
                )
            if not all(isinstance(item, dict) for item in raw_messages):
                raise ValueError("FlowPilot recovery provider messages are malformed")
            batches.append((events, tuple(raw_messages)))
        if not batches:
            raise ValueError("FlowPilot recovery contains no batches")
        return batches

    def _load_recovery_payload(
        self,
        manifest: dict[str, Any],
    ) -> tuple[list[Event], tuple[dict[str, Any], ...]]:
        if self._file_store is None:
            raise RuntimeError("FlowPilot DCS recovery requires a persistent FileStore")
        return self._load_recovery_batches(manifest)[0]

    def _recover_pending_dcs(
        self,
        *,
        manifest: dict[str, Any],
        context_cursor: str,
        context_digest_value: str,
        apply_events: Callable[
            [list[Event], tuple[dict[str, Any], ...]], tuple[str, str]
        ],
        reconciled: dict[str, Any],
    ) -> None:
        batches = self._load_recovery_batches(manifest)
        first_count = manifest.get("first_chunk_batch_count", 1)
        if not isinstance(first_count, int) or not 0 < first_count <= len(batches):
            raise ValueError("FlowPilot recovery chunk metadata is malformed")
        reference = self._require_dcs_reference()
        first_events = [
            event
            for batch_events, _messages in batches[:first_count]
            for event in batch_events
        ]
        first_messages = tuple(
            message
            for _events, batch_messages in batches[:first_count]
            for message in batch_messages
        )
        applied_cursor = manifest.get("applied_context_cursor")
        if not isinstance(applied_cursor, str):
            applied_cursor = manifest.get("last_event_id")
        already_acked = reconciled.get("base_context_cursor") == applied_cursor
        if not already_acked:
            context_cursor, context_digest_value = apply_events(
                first_events, first_messages
            )
            self._record_recovery_applied_context(
                manifest, context_cursor, context_digest_value
            )
        else:
            context_cursor = str(
                reconciled.get("base_context_cursor") or context_cursor
            )
            context_digest_value = str(
                reconciled.get("base_context_digest") or context_digest_value
            )
        first_seq = manifest.get("first_seq")
        last_seq = manifest.get("last_seq")
        delta_digest = manifest.get("delta_digest")
        if not isinstance(delta_digest, str):
            delta_digest = reference.delta_digest
        if not isinstance(first_seq, int) or not isinstance(last_seq, int):
            raise ValueError("FlowPilot recovery sequence metadata is malformed")
        if not isinstance(delta_digest, str):
            raise ValueError("FlowPilot recovery delta digest is malformed")
        if already_acked:
            pending = int(reconciled.get("pending_message_count", 0))
            if not pending:
                self._clear_delegation()
                self._clear_recovery_manifest(manifest)
                return
            response = self._request_json(
                "/flowpilot/v1/dcs/sync/next", reference.payload(), method="POST"
            )
        else:
            ack = self._request_json(
                "/flowpilot/v1/dcs/sync/ack",
                {
                    "protocol_version": "flowpilot-phase2-dcs-v2",
                    "reference": reference.payload(),
                    "first_seq": first_seq,
                    "last_seq": last_seq,
                    "delta_digest": delta_digest,
                    "new_context_cursor": context_cursor,
                    "new_context_digest": context_digest_value,
                },
                method="POST",
            )
            if not ack.get("pending_message_count"):
                self._clear_delegation()
                self._clear_recovery_manifest(manifest)
                return
            reference = FlowPilotDCSReference(
                job_id=reference.job_id,
                line_id=reference.line_id,
                context_epoch=reference.context_epoch,
                lease_id=reference.lease_id,
                base_context_cursor=context_cursor,
                delta_digest=_require_string(ack, "delta_digest"),
            )
            self._dcs_reference = reference
            response = self._request_json(
                "/flowpilot/v1/dcs/sync/next", reference.payload(), method="POST"
            )

        next_index = first_count
        while True:
            raw_messages = response.get("messages")
            if not isinstance(raw_messages, list) or not all(
                isinstance(item, dict) for item in raw_messages
            ):
                raise ValueError(
                    "FlowPilot recovery context sync messages are malformed"
                )
            redacted_messages = tuple(
                _redact_provider_message(item) for item in raw_messages
            )
            consumed: list[tuple[list[Event], tuple[dict[str, Any], ...]]] = []
            offset = 0
            while next_index + len(consumed) < len(batches):
                batch = batches[next_index + len(consumed)]
                expected = batch[1]
                end = offset + len(expected)
                if end > len(redacted_messages) or (
                    redacted_messages[offset:end] != expected
                ):
                    break
                consumed.append(batch)
                offset = end
            if not consumed or offset != len(redacted_messages):
                raise ValueError(
                    "FlowPilot recovery differs from persisted Agent history"
                )
            events = [event for batch_events, _ in consumed for event in batch_events]
            self._sync_response = response
            self._sync_reference = reference
            self.persist_recovery_record(
                messages=tuple(raw_messages),
                events=events,
                pending_batches=tuple(batches[next_index:]),
                first_chunk_batch_count=len(consumed),
            )
            updated_manifest = self._read_recovery_manifest()
            if updated_manifest is None:
                raise RuntimeError("FlowPilot recovery manifest disappeared")
            context_cursor, context_digest_value = apply_events(
                events, tuple(raw_messages)
            )
            self._record_recovery_applied_context(
                updated_manifest, context_cursor, context_digest_value
            )
            ack = self._request_json(
                "/flowpilot/v1/dcs/sync/ack",
                {
                    "protocol_version": "flowpilot-phase2-dcs-v2",
                    "reference": reference.payload(),
                    "first_seq": response.get("first_seq"),
                    "last_seq": response.get("last_seq"),
                    "delta_digest": response.get("delta_digest"),
                    "new_context_cursor": context_cursor,
                    "new_context_digest": context_digest_value,
                },
                method="POST",
            )
            next_index += len(consumed)
            if not ack.get("pending_message_count"):
                self._clear_delegation()
                self._clear_recovery_manifest(manifest)
                return
            reference = FlowPilotDCSReference(
                job_id=reference.job_id,
                line_id=reference.line_id,
                context_epoch=reference.context_epoch,
                lease_id=reference.lease_id,
                base_context_cursor=context_cursor,
                delta_digest=_require_string(response, "wal_delta_digest"),
            )
            self._dcs_reference = reference
            response = self._request_json(
                "/flowpilot/v1/dcs/sync/next", reference.payload(), method="POST"
            )

    def register(
        self,
        *,
        context_sequence: int = 0,
        base_context_cursor: str = "root",
        context_digest: str | None = None,
        apply_recovery: Callable[
            [list[Event], tuple[dict[str, Any], ...]], tuple[str, str]
        ]
        | None = None,
    ) -> None:
        """Register or resume the line before any LLM request."""
        self._post_control(
            "/flowpilot/v1/jobs",
            {
                "job_id": self.config.job_id,
                "root_conversation_id": self.config.root_conversation_id,
                "deployment_id": self.config.deployment_id,
                "namespace_id": self.config.namespace_id,
            },
        )
        digest = context_digest or hashlib.sha256(b"[]").hexdigest()
        line_payload = {
            "job_id": self.config.job_id,
            "line_id": self.config.line_id,
            "context_epoch": self.context_epoch,
            "context_sequence": context_sequence,
            "base_context_cursor": base_context_cursor,
            "context_digest": digest,
            "conversation_id": self.conversation_id,
            "parent_conversation_id": self.config.parent_conversation_id,
            "parent_line_id": self.config.parent_line_id,
            "spawn_id": self.config.spawn_id,
            "task_id": self.config.task_id,
            "agent_id": self.config.agent_id,
            "parent_action_id": self.config.parent_action_id,
        }
        tail = self._find_authoritative_tail()
        if tail is None:
            self._post_control("/flowpilot/v1/lines", line_payload)
        else:
            for field_name in (
                "conversation_id",
                "parent_conversation_id",
                "parent_line_id",
                "spawn_id",
            ):
                registered = tail.get(field_name)
                requested = line_payload.get(field_name)
                if registered is not None and registered != requested:
                    raise RuntimeError(
                        f"FlowPilot line identity mismatch: {field_name}"
                    )
            phase = _tail_phase(tail)
            version = tail.get("version")
            if phase not in {"EMPTY", "READY"} or not isinstance(version, int):
                raise RuntimeError(
                    "FlowPilot cannot resume a line with an in-progress tail"
                )
            self.tail_version = version
        self._registered = True
        if self.config.deferred_context_enabled:
            reconciled = self.reconcile_deferred_context(
                context_cursor=base_context_cursor,
                context_digest=digest,
            )
            manifest = self._read_recovery_manifest()
            if manifest is not None:
                if apply_recovery is None:
                    raise RuntimeError(
                        "FlowPilot pending DCS recovery needs OpenHands state"
                    )
                raw_reference = manifest.get("dcs_reference")
                if not isinstance(raw_reference, dict):
                    raise ValueError(
                        "FlowPilot recovery delegation reference is malformed"
                    )
                lease_id = raw_reference.get("lease_id")
                if not isinstance(lease_id, str):
                    raise ValueError("FlowPilot recovery lease ID is malformed")
                self._dcs_reference = FlowPilotDCSReference(
                    job_id=self.config.job_id,
                    line_id=self.config.line_id,
                    context_epoch=self.context_epoch,
                    lease_id=lease_id,
                    base_context_cursor=str(
                        reconciled.get("base_context_cursor")
                        or raw_reference.get("base_context_cursor")
                    ),
                    delta_digest=str(
                        reconciled.get("delta_digest")
                        or raw_reference.get("delta_digest")
                    ),
                )
                self._dcs_policy_version = int(manifest.get("policy_version", 0))
                self._dcs_base_context_sequence = int(
                    manifest.get("base_context_sequence", 0)
                )
                self._dcs_base_context_digest = manifest.get("base_context_digest")
                self._dcs_last_seq = int(manifest.get("last_seq", 0))
                self._recover_pending_dcs(
                    manifest=manifest,
                    context_cursor=base_context_cursor,
                    context_digest_value=digest,
                    apply_events=apply_recovery,
                    reconciled=reconciled,
                )
                return
            if reconciled.get("sync_required"):
                raise RuntimeError("FlowPilot WAL requires recovery metadata")
            if reconciled.get("status") in {"agent_stale", "context_diverged"}:
                raise RuntimeError(
                    "FlowPilot DCS state conflicts with the authoritative Agent history"
                )

    def _finish_line(self) -> None:
        """Retire the registered line when its conversation is explicitly closed."""
        if not self._registered:
            return
        with self._lock:
            if self._active_identity is not None or self._active_tools:
                raise RuntimeError("FlowPilot cannot finish while local work is active")
            version = self.tail_version
            identity = self._tool_identity
        tail = self._find_authoritative_tail()
        if tail is not None:
            if tail.get("version") != version or (
                identity is not None
                and tail.get("tail_request_id") != identity.tail_request_id
            ):
                raise RuntimeError("FlowPilot cannot finish a replaced tail")
            if _tail_phase(tail) not in {"EMPTY", "READY"}:
                raise RuntimeError("FlowPilot cannot finish an unresolved line")
            line_id = urllib.parse.quote(self.config.line_id, safe="")
            self._post_control(
                f"/flowpilot/v1/lines/{line_id}/finish",
                {
                    "job_id": self.config.job_id,
                    "line_id": self.config.line_id,
                    "expected_tail_version": tail["version"],
                    "tail_request_id": tail.get("tail_request_id"),
                },
            )
        self._registered = False

    def report_dependency(
        self, prerequisite_line_id: str, *, actual_wait: bool
    ) -> None:
        """Atomically report the lines this conversation is actually waiting on."""
        with self._lock:
            if actual_wait:
                self._waiting_on_lines.add(prerequisite_line_id)
            else:
                self._waiting_on_lines.discard(prerequisite_line_id)
            self._dependency_version += 1
            version = self._dependency_version
            prerequisites = tuple(sorted(self._waiting_on_lines))
        self._request_json(
            f"/flowpilot/v1/lines/{urllib.parse.quote(self.config.line_id, safe='')}"
            "/dependencies",
            {
                "protocol_version": "flowpilot-phase0-v2",
                "job_id": self.config.job_id,
                "line_id": self.config.line_id,
                "version": version,
                "prerequisite_line_ids": prerequisites,
            },
            method="PUT",
        )

    def begin_request(
        self,
        *,
        context_sequence: int,
        base_context_cursor: str,
        context_digest: str,
    ) -> FlowPilotRequestIdentity:
        with self._lock:
            identity = FlowPilotRequestIdentity(
                job_id=self.config.job_id,
                line_id=self.config.line_id,
                tail_request_id=str(uuid.uuid4()),
                request_id=str(uuid.uuid4()),
                llm_call_id=str(uuid.uuid4()),
                attempt=1,
                tail_version=self.tail_version,
                context_epoch=self.context_epoch,
                context_sequence=context_sequence,
                base_context_cursor=base_context_cursor,
                context_digest=context_digest,
                conversation_id=self.conversation_id,
                parent_conversation_id=self.config.parent_conversation_id,
                parent_line_id=self.config.parent_line_id,
                spawn_id=self.config.spawn_id,
                deployment_id=self.config.deployment_id,
                namespace_id=self.config.namespace_id,
            )
            self._active_identity = identity
            self._transport_attempts = 0
            self._gateway_reuse_policy = None
            self._gateway_decisions.clear()
            self._gateway_batches.clear()
            self._gateway_response_deferred = False
            return identity

    def configure_gateway_reuse(
        self,
        *,
        deferred: bool,
        api_kind: Literal["chat", "responses"],
        tool_schema_digests: dict[str, str] | None = None,
    ) -> None:
        if not self.config.exact_reuse_enabled:
            return
        now = datetime.now(UTC)
        self._gateway_reuse_policy = {
            "allowed_tool_names": list(self.config.reusable_web_tools),
            "tool_schema_digests": tool_schema_digests or {},
            "reuse_protocol_version": self.config.reuse_protocol_version,
            "deferred": deferred,
            "policy_version": self._dcs_policy_version + 1,
            "expected_policy_version": self._dcs_policy_version,
            "lease_id": str(uuid.uuid4()),
            "lease_seconds": self.config.delegation_lease_seconds,
            "max_messages": self.config.deferred_max_messages,
            "max_bytes": self.config.deferred_max_bytes,
            "max_internal_continuations": self.config.max_internal_continuations,
            "delta_ttl_seconds": self.config.deferred_delta_ttl_seconds,
            "locale": self.config.locale,
            "language": self.config.language,
            "region": self.config.region,
            "safe_search_policy": self.config.safe_search_policy,
            "time_sensitivity_class": self.config.time_sensitivity_class,
            "data_source_constraints": list(self.config.data_source_constraints),
            "output_budget_bytes": self.config.reuse_output_budget_bytes,
            "api_kind": api_kind,
            "issued_at": now.isoformat(),
        }

    def prepare_attempt(self) -> dict[str, str]:
        """Bind one provider transport invocation to the logical request."""
        with self._lock:
            identity = self._active_identity
            if identity is None:
                raise RuntimeError("FlowPilot has no active logical request")
            if self._transport_attempts:
                identity = replace(
                    identity,
                    attempt=identity.attempt + 1,
                    llm_call_id=str(uuid.uuid4()),
                )
                self._active_identity = identity
            self._transport_attempts += 1
            headers = identity.headers(self.config.api_key)
            if self._gateway_reuse_policy is not None:
                headers["x-flowpilot-reuse-policy"] = json.dumps(
                    self._gateway_reuse_policy,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            return headers

    def accept_gateway_response(self, response: Any) -> None:
        """Accept Scheduler-owned reuse metadata before tail commit."""
        raw = getattr(response, "raw_response", response)
        metadata = raw.get("flowpilot") if hasattr(raw, "get") else None
        if not isinstance(metadata, dict):
            return
        with self._lock:
            identity = self._active_identity
        if identity is None:
            raise RuntimeError("FlowPilot has no active request for response metadata")
        for item in metadata.get("decisions", []):
            if not isinstance(item, dict):
                raise RuntimeError("FlowPilot reuse metadata is malformed")
            item_identity = item.get("identity")
            if not isinstance(item_identity, dict):
                raise RuntimeError("FlowPilot reuse identity is missing")
            if (
                item_identity.get("job_id") != identity.job_id
                or item_identity.get("line_id") != identity.line_id
            ):
                raise RuntimeError("FlowPilot reuse identity does not match the line")
            tool_call_id = item_identity.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                raise RuntimeError("FlowPilot reuse tool_call_id is missing")
            with self._lock:
                self._gateway_decisions[tool_call_id] = item
        batches = metadata.get("batches")
        if batches is not None:
            if not isinstance(batches, list) or not all(
                isinstance(item, dict) for item in batches
            ):
                raise RuntimeError("FlowPilot gateway DCS batches are malformed")
            with self._lock:
                self._gateway_batches = list(batches)
                self._gateway_response_deferred = (
                    metadata.get("response_deferred") is True
                )
                reason = metadata.get("barrier_reason")
                self._dcs_required_sync_reason = (
                    reason if isinstance(reason, str) else None
                )
        policy_version = metadata.get("policy_version")
        if isinstance(policy_version, int):
            with self._lock:
                self._dcs_policy_version = policy_version
        final = metadata.get("final_identity")
        if final is not None:
            if not isinstance(final, dict):
                raise RuntimeError("FlowPilot final identity is malformed")
            if (
                final.get("job_id") != identity.job_id
                or final.get("line_id") != identity.line_id
                or final.get("conversation_id") != identity.conversation_id
                or final.get("origin") != "scheduler_delegated"
                or not isinstance(final.get("delegation_lease_id"), str)
            ):
                raise RuntimeError("FlowPilot continuation identity is invalid")
            with self._lock:
                self._active_identity = replace(
                    identity,
                    request_id=str(final["request_id"]),
                    tail_request_id=str(final["tail_request_id"]),
                    llm_call_id=str(final["llm_call_id"]),
                    attempt=int(final["attempt"]),
                    tail_version=int(final["expected_tail_version"]),
                    context_sequence=int(final["context_sequence"]),
                    base_context_cursor=str(final["base_context_cursor"]),
                    context_digest=str(final["context_digest"]),
                    origin="scheduler_delegated",
                    delegation_lease_id=str(final["delegation_lease_id"]),
                )
        reference = metadata.get("dcs_reference")
        if isinstance(reference, dict):
            with self._lock:
                self._dcs_reference = FlowPilotDCSReference(
                    job_id=str(reference["job_id"]),
                    line_id=str(reference["line_id"]),
                    context_epoch=int(reference["context_epoch"]),
                    lease_id=str(reference["lease_id"]),
                    base_context_cursor=str(reference["base_context_cursor"]),
                    delta_digest=str(reference["delta_digest"]),
                )
                self._dcs_base_context_sequence = identity.context_sequence
                self._dcs_base_context_digest = identity.context_digest
                self._dcs_last_seq = int(metadata.get("delta_seq", 0))

    def commit_request(self, identity: FlowPilotRequestIdentity) -> None:
        tail = self._fetch_authoritative_tail()
        if (
            tail.get("tail_request_id") != identity.tail_request_id
            or tail.get("llm_call_id") != identity.llm_call_id
            or _tail_phase(tail) not in {"BLOCKED", "READY"}
        ):
            raise RuntimeError(
                "FlowPilot did not commit the completed LLM call as authoritative"
            )
        version = tail.get("version")
        if not isinstance(version, int) or version <= identity.tail_version:
            raise RuntimeError(
                "FlowPilot returned an invalid authoritative tail version"
            )
        with self._lock:
            if self._active_identity == identity:
                self.tail_version = version
                self._active_identity = None
                self._tool_identity = identity

    def begin_delegated_request(self) -> FlowPilotRequestIdentity:
        reference = self._require_dcs_reference()
        with self._lock:
            identity = FlowPilotRequestIdentity(
                job_id=self.config.job_id,
                line_id=self.config.line_id,
                tail_request_id=str(uuid.uuid4()),
                request_id=str(uuid.uuid4()),
                llm_call_id=str(uuid.uuid4()),
                attempt=1,
                tail_version=self.tail_version,
                context_epoch=reference.context_epoch,
                context_sequence=self._dcs_base_context_sequence + self._dcs_last_seq,
                base_context_cursor=reference.base_context_cursor,
                context_digest=reference.delta_digest,
                conversation_id=self.conversation_id,
                parent_conversation_id=self.config.parent_conversation_id,
                parent_line_id=self.config.parent_line_id,
                spawn_id=self.config.spawn_id,
                deployment_id=self.config.deployment_id,
                namespace_id=self.config.namespace_id,
                origin="scheduler_delegated",
                delegation_lease_id=reference.lease_id,
            )
            self._active_identity = identity
            self._transport_attempts = 0
            return identity

    def abort_request(self, identity: FlowPilotRequestIdentity) -> None:
        try:
            tail = self._fetch_authoritative_tail()
        except Exception:
            logger.warning("FlowPilot tail reconciliation failed", exc_info=True)
            tail = None
        with self._lock:
            if self._active_identity == identity:
                if tail is not None:
                    version = tail.get("version")
                    if isinstance(version, int):
                        self.tail_version = version
                self._active_identity = None

    def grant_delegation(
        self,
        identity: FlowPilotRequestIdentity,
        *,
        api_kind: Literal["chat", "responses"],
        request_snapshot: dict[str, Any],
    ) -> FlowPilotDCSReference:
        """Grant FlowPilot a bounded single-writer lease for exact-hit DCS."""
        if not self.config.deferred_context_enabled:
            raise RuntimeError("FlowPilot deferred context is disabled")
        now = datetime.now(UTC)
        lease_id = str(uuid.uuid4())
        policy_version = self._dcs_policy_version + 1
        policy_payload = {
            "protocol_version": "flowpilot-phase2-dcs-v2",
            "policy_version": policy_version,
            "expected_policy_version": self._dcs_policy_version,
            "lease_id": lease_id,
            "job_id": identity.job_id,
            "line_id": identity.line_id,
            "context_epoch": identity.context_epoch,
            "base_context_cursor": identity.base_context_cursor,
            "base_context_digest": identity.context_digest,
            "issued_at": now.isoformat(),
            "expires_at": (
                now + timedelta(seconds=self.config.delegation_lease_seconds)
            ).isoformat(),
            "allowed_tool_names": list(self.config.reusable_web_tools),
            "max_messages": self.config.deferred_max_messages,
            "max_bytes": self.config.deferred_max_bytes,
            "max_internal_continuations": (self.config.max_internal_continuations),
            "delta_ttl_seconds": self.config.deferred_delta_ttl_seconds,
            "api_kind": api_kind,
            "request_snapshot": request_snapshot,
        }
        try:
            response = self._request_json(
                "/flowpilot/v1/dcs/delegations", policy_payload, method="POST"
            )
        except Exception:
            response = self._reconcile_payload(
                context_cursor=identity.base_context_cursor,
                context_digest=identity.context_digest,
            )
            if (
                response.get("lease_id") != lease_id
                or response.get("policy_version") != policy_version
                or response.get("state") != "open"
            ):
                raise
        reference = FlowPilotDCSReference(
            job_id=identity.job_id,
            line_id=identity.line_id,
            context_epoch=identity.context_epoch,
            lease_id=lease_id,
            base_context_cursor=identity.base_context_cursor,
            delta_digest=_require_string(response, "delta_digest"),
        )
        with self._lock:
            self._dcs_policy_version = policy_version
            self._dcs_reference = reference
            self._dcs_base_context_sequence = identity.context_sequence
            self._dcs_base_context_digest = identity.context_digest
            self._dcs_last_seq = 0
            self._dcs_required_sync_reason = None
        return reference

    def release_delegation(self) -> None:
        """Release an empty lease before returning control to the Agent."""
        reference = self._require_dcs_reference()
        response = self._request_json(
            "/flowpilot/v1/dcs/delegations/release",
            reference.payload(),
            method="POST",
        )
        if response.get("state") != "aborted" or response.get("pending_message_count"):
            raise RuntimeError("FlowPilot did not release the empty delegation")
        self._clear_delegation()

    def reconcile_active_delegation(self) -> dict[str, Any]:
        """Refresh the local reference from the durable DCS WAL."""
        reference = self._require_dcs_reference()
        with self._lock:
            base_digest = self._dcs_base_context_digest
        if base_digest is None:
            raise RuntimeError("FlowPilot delegation has no base context digest")
        response = self._reconcile_payload(
            context_cursor=reference.base_context_cursor,
            context_digest=base_digest,
        )
        lease_id = response.get("lease_id")
        if lease_id != reference.lease_id:
            raise RuntimeError("FlowPilot reconciliation returned a different writer")
        state = response.get("state")
        if state not in {"open", "syncing"}:
            raise RuntimeError(
                f"FlowPilot active delegation reconciled to invalid state {state}"
            )
        updated = FlowPilotDCSReference(
            job_id=reference.job_id,
            line_id=reference.line_id,
            context_epoch=reference.context_epoch,
            lease_id=reference.lease_id,
            base_context_cursor=_require_string(response, "base_context_cursor"),
            delta_digest=_require_string(response, "delta_digest"),
        )
        last_seq = response.get("last_seq")
        if not isinstance(last_seq, int) or last_seq < 0:
            raise ValueError("FlowPilot reconciliation returned an invalid last_seq")
        reason = response.get("barrier_reason") if state == "syncing" else None
        if reason is not None and not isinstance(reason, str):
            raise ValueError("FlowPilot reconciliation returned an invalid barrier")
        with self._lock:
            self._dcs_reference = updated
            self._dcs_last_seq = last_seq
            self._dcs_required_sync_reason = reason
        return response

    def append_deferred_context(
        self,
        *,
        expected_last_seq: int,
        parent_llm_call_id: str,
        messages: list[dict[str, Any]],
        tool_call_ids: list[str],
        resolution_receipts: list[str],
        result_digests: list[str],
    ) -> FlowPilotDCSReference:
        """Append one complete provider-valid assistant/Tool batch to the WAL."""
        reference = self._require_dcs_reference()
        response = self._request_json(
            "/flowpilot/v1/dcs/deltas/append",
            {
                "protocol_version": "flowpilot-phase2-dcs-v2",
                "reference": reference.payload(),
                "expected_last_seq": expected_last_seq,
                "parent_llm_call_id": parent_llm_call_id,
                "messages": messages,
                "tool_call_ids": tool_call_ids,
                "resolution_receipts": resolution_receipts,
                "result_digests": result_digests,
            },
            method="POST",
        )
        updated = FlowPilotDCSReference(
            job_id=reference.job_id,
            line_id=reference.line_id,
            context_epoch=reference.context_epoch,
            lease_id=reference.lease_id,
            base_context_cursor=reference.base_context_cursor,
            delta_digest=_require_string(response, "delta_digest"),
        )
        last_seq = response.get("last_seq")
        if not isinstance(last_seq, int) or last_seq <= expected_last_seq:
            raise ValueError("FlowPilot append response has an invalid last_seq")
        state = response.get("state")
        if state not in {"open", "syncing"}:
            raise ValueError("FlowPilot append response has an invalid state")
        sync_reason = response.get("barrier_reason") if state == "syncing" else None
        if sync_reason is not None and not isinstance(sync_reason, str):
            raise ValueError("FlowPilot append response has an invalid barrier reason")
        with self._lock:
            self._dcs_reference = updated
            self._dcs_last_seq = last_seq
            self._dcs_required_sync_reason = sync_reason
        return updated

    def resolve_deferred_reuse(self, action: ActionEvent) -> dict[str, Any]:
        """Resolve an exact hit under DCS without executing or injecting a Tool."""
        if action.action is None:
            raise ValueError("FlowPilot cannot reuse an action without arguments")
        with self._lock:
            identity = self._tool_identity
        if identity is None:
            raise RuntimeError("FlowPilot has no completed LLM Tool Call identity")
        reference = self._require_dcs_reference()
        reuse_identity = self._reuse_identity(identity, action)
        reuse = {
            "protocol_version": "flowpilot-phase1-reuse-v3",
            "identity": reuse_identity,
            "tool_name": action.tool_name,
            "arguments": _provider_tool_arguments(action),
            "scope": {
                "locale": self.config.locale,
                "language": self.config.language,
                "region": self.config.region,
                "safe_search_policy": self.config.safe_search_policy,
                "time_sensitivity_class": self.config.time_sensitivity_class,
                "data_source_constraints": list(self.config.data_source_constraints),
            },
            "output_budget_bytes": self.config.reuse_output_budget_bytes,
        }
        decision = self._request_json(
            "/flowpilot/v1/dcs/reuse/resolve",
            {
                "protocol_version": "flowpilot-phase2-dcs-v2",
                "reuse": reuse,
                "delegation": reference.payload(),
            },
            method="POST",
        )
        binding_id = decision.get("binding_id")
        deadline = time.monotonic() + self.config.reuse_wait_timeout
        try:
            while (
                decision.get("decision") == "defer_wait_for_inflight"
                and isinstance(binding_id, str)
                and time.monotonic() < deadline
            ):
                time.sleep(self.config.reuse_poll_interval)
                decision = self._request_json(
                    "/flowpilot/v1/dcs/reuse/bindings/poll",
                    {
                        "protocol_version": "flowpilot-phase2-dcs-v2",
                        "binding_id": binding_id,
                        "reuse": reuse,
                        "delegation": reference.payload(),
                    },
                    method="POST",
                )
        except Exception:
            if isinstance(binding_id, str):
                self._cancel_follower(binding_id, reuse_identity)
            raise
        if (
            isinstance(binding_id, str)
            and decision.get("decision") != "defer_with_cached_result"
        ):
            self._cancel_follower(binding_id, reuse_identity)
        return decision

    def deferred_observation(
        self,
        observation_type: type[Observation],
        decision: dict[str, Any],
    ) -> Observation:
        """Reconstruct a validated Observation without recording local execution."""
        if decision.get("decision") != "defer_with_cached_result":
            raise ValueError("FlowPilot decision is not a completed deferred result")
        result = decision.get("result")
        provider_content = decision.get("provider_content")
        if not isinstance(result, dict) or not isinstance(provider_content, str):
            raise ValueError("FlowPilot deferred result is malformed")
        if not isinstance(decision.get("resolution_receipt"), str) or not isinstance(
            decision.get("result_digest"), str
        ):
            raise ValueError("FlowPilot deferred result proof is malformed")
        observation = observation_type.model_validate(
            json.loads(canonical_json(result))
        )
        return _with_provider_content(observation, provider_content)

    def prepare_internal_continuation(self, parent_llm_call_id: str) -> dict[str, Any]:
        """Return the scheduler-built request body without modifying it locally."""
        response = self._request_json(
            "/flowpilot/v1/dcs/continuations",
            {
                "protocol_version": "flowpilot-phase2-dcs-v2",
                "reference": self._require_dcs_reference().payload(),
                "parent_llm_call_id": parent_llm_call_id,
            },
            method="POST",
        )
        body = response.get("body")
        if not isinstance(body, dict):
            raise ValueError("FlowPilot continuation body is malformed")
        return body

    def synchronize_deferred_context(
        self,
        *,
        barrier_reason: str,
        apply_atomically: Callable[[tuple[dict[str, Any], ...]], tuple[str, str]],
        parent_llm_call_id: str | None = None,
        barrier_messages: tuple[dict[str, Any], ...] = (),
        pending_local_tool_call_ids: tuple[str, ...] = (),
    ) -> None:
        """Apply every sync chunk locally, ACKing only after atomic application."""
        reference = self._require_dcs_reference()
        sync_payload = {
            "protocol_version": "flowpilot-phase2-dcs-v2",
            "reference": reference.payload(),
            "barrier_reason": barrier_reason,
            "parent_llm_call_id": parent_llm_call_id,
            "barrier_messages": list(barrier_messages),
            "pending_local_tool_call_ids": list(pending_local_tool_call_ids),
        }
        try:
            response = self._request_json(
                "/flowpilot/v1/dcs/sync", sync_payload, method="POST"
            )
        except Exception:
            reference, response = self._recover_sync_start()
        while True:
            raw_messages = response.get("messages")
            if not isinstance(raw_messages, list) or not all(
                isinstance(item, dict) for item in raw_messages
            ):
                raise ValueError("FlowPilot context sync messages are malformed")
            messages = tuple(raw_messages)
            self._sync_response = response
            self._sync_reference = reference
            cursor, context_digest = apply_atomically(messages)
            if not isinstance(cursor, str) or not cursor:
                raise ValueError(
                    "atomic context application returned an invalid cursor"
                )
            if not isinstance(context_digest, str) or len(context_digest) != 64:
                raise ValueError(
                    "atomic context application returned an invalid digest"
                )
            self._record_recovery_applied_context(
                self._read_recovery_manifest(), cursor, context_digest
            )
            ack_payload = {
                "protocol_version": "flowpilot-phase2-dcs-v2",
                "reference": reference.payload(),
                "first_seq": response.get("first_seq"),
                "last_seq": response.get("last_seq"),
                "delta_digest": response.get("delta_digest"),
                "new_context_cursor": cursor,
                "new_context_digest": context_digest,
            }
            ack = self._request_json(
                "/flowpilot/v1/dcs/sync/ack", ack_payload, method="POST"
            )
            if not ack.get("pending_message_count"):
                self._clear_delegation()
                self._clear_recovery_manifest(self._read_recovery_manifest())
                self._sync_response = None
                self._sync_reference = None
                return
            reference = FlowPilotDCSReference(
                job_id=reference.job_id,
                line_id=reference.line_id,
                context_epoch=reference.context_epoch,
                lease_id=reference.lease_id,
                base_context_cursor=cursor,
                delta_digest=_require_string(response, "wal_delta_digest"),
            )
            with self._lock:
                self._dcs_reference = reference
            response = self._request_json(
                "/flowpilot/v1/dcs/sync/next",
                reference.payload(),
                method="POST",
            )

    async def asynchronize_deferred_context(
        self,
        *,
        barrier_reason: str,
        apply_atomically: Callable[[tuple[dict[str, Any], ...]], tuple[str, str]],
        parent_llm_call_id: str | None = None,
        barrier_messages: tuple[dict[str, Any], ...] = (),
        pending_local_tool_call_ids: tuple[str, ...] = (),
    ) -> None:
        """Async sync that keeps local event application on the caller's thread."""
        reference = self._require_dcs_reference()
        sync_payload = {
            "protocol_version": "flowpilot-phase2-dcs-v2",
            "reference": reference.payload(),
            "barrier_reason": barrier_reason,
            "parent_llm_call_id": parent_llm_call_id,
            "barrier_messages": list(barrier_messages),
            "pending_local_tool_call_ids": list(pending_local_tool_call_ids),
        }
        try:
            response = await asyncio.to_thread(
                self._request_json,
                "/flowpilot/v1/dcs/sync",
                sync_payload,
                method="POST",
            )
        except Exception:
            reference, response = await asyncio.to_thread(self._recover_sync_start)
        while True:
            raw_messages = response.get("messages")
            if not isinstance(raw_messages, list) or not all(
                isinstance(item, dict) for item in raw_messages
            ):
                raise ValueError("FlowPilot context sync messages are malformed")
            self._sync_response = response
            self._sync_reference = reference
            cursor, context_digest = apply_atomically(tuple(raw_messages))
            if not isinstance(cursor, str) or not cursor:
                raise ValueError(
                    "atomic context application returned an invalid cursor"
                )
            if not isinstance(context_digest, str) or len(context_digest) != 64:
                raise ValueError(
                    "atomic context application returned an invalid digest"
                )
            self._record_recovery_applied_context(
                self._read_recovery_manifest(), cursor, context_digest
            )
            ack = await asyncio.to_thread(
                self._request_json,
                "/flowpilot/v1/dcs/sync/ack",
                {
                    "protocol_version": "flowpilot-phase2-dcs-v2",
                    "reference": reference.payload(),
                    "first_seq": response.get("first_seq"),
                    "last_seq": response.get("last_seq"),
                    "delta_digest": response.get("delta_digest"),
                    "new_context_cursor": cursor,
                    "new_context_digest": context_digest,
                },
                method="POST",
            )
            if not ack.get("pending_message_count"):
                self._clear_delegation()
                self._clear_recovery_manifest(self._read_recovery_manifest())
                self._sync_response = None
                self._sync_reference = None
                return
            reference = FlowPilotDCSReference(
                job_id=reference.job_id,
                line_id=reference.line_id,
                context_epoch=reference.context_epoch,
                lease_id=reference.lease_id,
                base_context_cursor=cursor,
                delta_digest=_require_string(response, "wal_delta_digest"),
            )
            with self._lock:
                self._dcs_reference = reference
            response = await asyncio.to_thread(
                self._request_json,
                "/flowpilot/v1/dcs/sync/next",
                reference.payload(),
                method="POST",
            )

    def reconcile_deferred_context(
        self, *, context_cursor: str, context_digest: str
    ) -> dict[str, Any]:
        """Compare the local authoritative cursor with the durable DCS WAL."""
        return self._reconcile_payload(
            context_cursor=context_cursor,
            context_digest=context_digest,
        )

    def _reconcile_payload(
        self, *, context_cursor: str, context_digest: str
    ) -> dict[str, Any]:
        return self._request_json(
            "/flowpilot/v1/dcs/reconcile",
            {
                "protocol_version": "flowpilot-phase2-dcs-v2",
                "job_id": self.config.job_id,
                "line_id": self.config.line_id,
                "context_epoch": self.context_epoch,
                "context_cursor": context_cursor,
                "context_digest": context_digest,
            },
            method="POST",
        )

    def _recover_sync_start(
        self,
    ) -> tuple[FlowPilotDCSReference, dict[str, Any]]:
        reconciled = self.reconcile_active_delegation()
        reference = self._require_dcs_reference()
        if not reconciled.get("sync_required"):
            raise RuntimeError("FlowPilot has no durable context to recover")
        if reconciled.get("state") == "syncing":
            response = self._request_json(
                "/flowpilot/v1/dcs/sync/next",
                reference.payload(),
                method="POST",
            )
        else:
            response = self._request_json(
                "/flowpilot/v1/dcs/sync",
                {
                    "protocol_version": "flowpilot-phase2-dcs-v2",
                    "reference": reference.payload(),
                    "barrier_reason": "failure",
                    "parent_llm_call_id": None,
                    "barrier_messages": [],
                    "pending_local_tool_call_ids": [],
                },
                method="POST",
            )
        return reference, response

    def _clear_delegation(self) -> None:
        with self._lock:
            self._dcs_reference = None
            self._dcs_base_context_digest = None
            self._dcs_last_seq = 0
            self._dcs_required_sync_reason = None
            self._gateway_batches.clear()
            self._gateway_response_deferred = False
            self._gateway_decisions.clear()

    def _require_dcs_reference(self) -> FlowPilotDCSReference:
        with self._lock:
            reference = self._dcs_reference
        if reference is None:
            raise RuntimeError("FlowPilot has no active delegation lease")
        return reference

    @property
    def has_pending_deferred_context(self) -> bool:
        with self._lock:
            return self._dcs_reference is not None and self._dcs_last_seq > 0

    @property
    def deferred_last_seq(self) -> int:
        with self._lock:
            return self._dcs_last_seq

    @property
    def deferred_sync_reason(self) -> str | None:
        with self._lock:
            return self._dcs_required_sync_reason

    @property
    def active_identity(self) -> FlowPilotRequestIdentity | None:
        with self._lock:
            return self._active_identity

    @property
    def gateway_batches(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._gateway_batches)

    @property
    def gateway_response_deferred(self) -> bool:
        """Whether the current response is already included in the DCS batches."""
        with self._lock:
            return self._gateway_response_deferred

    @property
    def gateway_reuse_configured(self) -> bool:
        """Whether this request is using Scheduler-owned reuse control."""
        with self._lock:
            return self._gateway_reuse_policy is not None

    @property
    def tool_identity(self) -> FlowPilotRequestIdentity | None:
        with self._lock:
            return self._tool_identity

    def tool_start(
        self, action: ActionEvent
    ) -> tuple[FlowPilotRequestIdentity, int] | None:
        """Emit START immediately before a real local executor invocation."""
        with self._lock:
            identity = self._tool_identity
            if identity is None:
                return None
            attempt = self._attempts.get(action.tool_call_id, 0) + 1
            self._attempts[action.tool_call_id] = attempt
        common = self._tool_common(identity, action, attempt)
        with self._lock:
            credentials = self._leader_credentials.get(action.id)
        if credentials is not None:
            common.update(
                {
                    key: credentials[key]
                    for key in (
                        "binding_id",
                        "input_digest",
                        "adapter_id",
                        "adapter_version",
                        "result_schema_version",
                        "executor_kind",
                    )
                }
            )
            common["reuse_receipt_version"] = "flowpilot-execution-v1"
        event_id = str(uuid.uuid4())
        if credentials is not None:
            credentials["start_event_id"] = event_id
            credentials["execution_attempt"] = attempt
        self._post_tool_event(
            {
                **common,
                "event_id": event_id,
                "sequence": 1,
                "event_kind": "start",
            }
        )
        with self._lock:
            self._active_tools[action.id] = (identity, attempt, time.monotonic())
        self._report_leader_progress(action, identity)
        return identity, attempt

    def _report_leader_progress(
        self, action: ActionEvent, identity: FlowPilotRequestIdentity
    ) -> None:
        if not self.config.semantic_reuse_enabled:
            return
        with self._lock:
            leader = self._leader_bindings.get(action.id)
        if leader is None or leader[0] != identity:
            return
        try:
            self._request_json(
                f"/flowpilot/v1/reuse/bindings/{leader[1]}/progress",
                {
                    "protocol_version": self.config.reuse_protocol_version,
                    "binding_id": leader[1],
                    "identity": self._reuse_identity(identity, action),
                    "sequence": 1,
                    "observed_at": datetime.now(UTC).isoformat(),
                },
                method="POST",
            )
        except Exception:
            logger.warning("FlowPilot leader progress report failed", exc_info=True)

    def resolve_reuse(
        self,
        action: ActionEvent,
        observation_type: type[Observation] | None,
        *,
        input_schema_digest: str | None = None,
    ) -> Observation | None:
        """Return a validated reused observation, or None to execute locally."""
        if (
            not self.config.exact_reuse_enabled
            or action.tool_name not in self.config.reusable_web_tools
            or observation_type is None
            or action.action is None
        ):
            return None
        with self._lock:
            identity = self._tool_identity
            gateway_decision = self._gateway_decisions.pop(action.tool_call_id, None)
        if identity is None:
            return None
        # SSE has no gateway decision envelope. Resolve those calls at the
        # Tool boundary; an explicit gateway decision still avoids a second RPC.
        reuse_identity = self._reuse_identity(identity, action)
        payload: dict[str, Any] = {
            "protocol_version": self.config.reuse_protocol_version,
            "identity": reuse_identity,
            "tool_name": action.tool_name,
            "arguments": _execution_tool_arguments(action),
            "scope": {
                "locale": self.config.locale,
                "language": self.config.language,
                "region": self.config.region,
                "safe_search_policy": self.config.safe_search_policy,
                "time_sensitivity_class": self.config.time_sensitivity_class,
                "data_source_constraints": list(self.config.data_source_constraints),
            },
            "output_budget_bytes": self.config.reuse_output_budget_bytes,
            "input_schema_digest": input_schema_digest,
        }
        cancel_event = threading.Event()
        with self._lock:
            self._reuse_cancellations[action.id] = cancel_event
        try:
            if cancel_event.is_set():
                raise asyncio.CancelledError()
            actual_digest = payload_digest(
                canonical_input(action.tool_name, payload["arguments"])
            )
            decision = gateway_decision or self._request_json(
                "/flowpilot/v1/reuse/resolve", payload, method="POST"
            )
            if gateway_decision is not None:
                expected_identity = {**reuse_identity, "action_id": None}
                if gateway_decision.get("identity") != expected_identity:
                    raise ValueError("Gateway decision ToolCallRef mismatch")
            if decision.get("decision") != "execute_locally" and (
                decision.get("input_digest") != actual_digest
                or decision.get("input_schema_digest") != input_schema_digest
            ):
                self._reject_gateway_binding(decision, reuse_identity)
                decision = self._request_json(
                    "/flowpilot/v1/reuse/resolve", payload, method="POST"
                )
                if decision.get("decision") != "execute_locally" and (
                    decision.get("input_digest") != actual_digest
                    or decision.get("input_schema_digest") != input_schema_digest
                ):
                    raise ValueError(
                        "Runtime input digest does not match reuse adapter"
                    )
            # A lease expiry releases a follower. Re-resolve once so the
            # caller can become the new leader, while bounding control-plane
            # waits if the replacement binding is also unavailable.
            for retry in range(2):
                if cancel_event.is_set():
                    raise asyncio.CancelledError()
                kind = decision.get("decision")
                binding_id = decision.get("binding_id")
                if kind == "sync_and_execute_as_leader" and isinstance(binding_id, str):
                    with self._lock:
                        self._leader_bindings[action.id] = (identity, binding_id)
                        self._leader_credentials[action.id] = dict(decision)
                    return None
                if kind not in {
                    "wait_and_sync_reused_result",
                    "defer_wait_for_inflight",
                } or not isinstance(binding_id, str):
                    break
                with self._lock:
                    self._waiting_reuses[action.id] = (binding_id, reuse_identity)
                decision = self._wait_for_reuse(
                    binding_id, reuse_identity, cancel_event=cancel_event
                )
                kind = decision.get("decision")
                if kind == "cancelled":
                    raise asyncio.CancelledError()
                if kind != "execute_locally" or retry == 1:
                    break
                decision = self._request_json(
                    "/flowpilot/v1/reuse/resolve", payload, method="POST"
                )
            if cancel_event.is_set():
                raise asyncio.CancelledError()
            kind = decision.get("decision")
            if kind not in {"sync_with_reused_result", "defer_with_cached_result"}:
                return None
            result = decision.get("result")
            provenance = decision.get("provenance")
            if not isinstance(result, dict) or not isinstance(provenance, dict):
                raise ValueError("FlowPilot reuse result is malformed")
            # OpenHands' discriminator validator removes `kind` from its input.
            # Preserve the committed wire value for digest validation and retries.
            observation = observation_type.model_validate(
                json.loads(canonical_json(result))
            )
            expiry = provenance.get("expires_at")
            if not isinstance(expiry, str) or datetime.fromisoformat(
                expiry
            ) <= datetime.now(UTC):
                logger.info("FlowPilot result expired before Observation delivery")
                return None
            if payload_digest(result) != provenance.get("result_digest"):
                raise ValueError("FlowPilot result digest mismatch")
            return _with_reuse_provenance(observation, result, provenance)
        except asyncio.CancelledError:
            with self._lock:
                waiting = self._waiting_reuses.get(action.id)
            if waiting is not None:
                self._cancel_follower(*waiting)
            raise
        except Exception:
            with self._lock:
                waiting = self._waiting_reuses.get(action.id)
            if waiting is not None:
                self._cancel_follower(*waiting)
            logger.warning(
                "FlowPilot exact reuse failed; executing Tool locally", exc_info=True
            )
            return None
        finally:
            with self._lock:
                self._reuse_cancellations.pop(action.id, None)
                self._waiting_reuses.pop(action.id, None)

    def _reject_gateway_binding(
        self, decision: dict[str, Any], identity: dict[str, str]
    ) -> None:
        binding = decision.get("binding_id")
        if not isinstance(binding, str):
            return
        if decision.get("decision") == "sync_and_execute_as_leader":
            self._request_json(
                f"/flowpilot/v1/reuse/bindings/{binding}/fail",
                {
                    "protocol_version": self.config.reuse_protocol_version,
                    "binding_id": binding,
                    "identity": identity,
                    "error_class": "InputChangedBeforeExecution",
                },
                method="POST",
            )
        else:
            self._cancel_follower(binding, identity)

    def tool_blocked(self, action: ActionEvent) -> None:
        """Report a policy-blocked call without claiming local execution."""
        with self._lock:
            identity = self._tool_identity
            if identity is None:
                return
            attempt = self._attempts.get(action.tool_call_id, 0) + 1
            self._attempts[action.tool_call_id] = attempt
        common = self._tool_common(identity, action, attempt)
        self._post_tool_event(
            {
                **common,
                "event_id": str(uuid.uuid4()),
                "sequence": 1,
                "event_kind": "blocked",
                "error_class": "HookBlocked",
            }
        )

    def tool_terminal(
        self,
        action: ActionEvent,
        *,
        token: tuple[FlowPilotRequestIdentity, int] | None,
        started: float,
        events: list[Event] | None = None,
        error: BaseException | None = None,
        cancelled: bool = False,
    ) -> None:
        if token is None:
            return
        identity, attempt = token
        with self._lock:
            active = self._active_tools.pop(action.id, None)
        if active is None or active[:2] != (identity, attempt):
            return
        common = self._tool_common(identity, action, attempt)
        latency = round((time.monotonic() - started) * 1000, 3)
        terminal: dict[str, Any]
        if cancelled:
            terminal = {"event_kind": "cancel", "error_class": "CancelledError"}
        elif error is not None:
            terminal = {"event_kind": "fail", "error_class": type(error).__name__}
        else:
            terminal = {
                "event_kind": "finish",
                "measured_latency_ms": latency,
                "result_size_bytes": _event_result_size(events or []),
            }
        event_id = str(uuid.uuid4())
        with self._lock:
            credentials = self._leader_credentials.get(action.id)
        if credentials is not None:
            common.update(
                {
                    key: credentials[key]
                    for key in (
                        "binding_id",
                        "input_digest",
                        "adapter_id",
                        "adapter_version",
                        "result_schema_version",
                        "executor_kind",
                    )
                }
            )
            common["reuse_receipt_version"] = "flowpilot-execution-v1"
            if events and isinstance(events[0], ObservationEvent):
                try:
                    result = events[0].observation.model_dump(mode="json")
                    credentials["result"] = result
                    credentials["result_digest"] = payload_digest(result)
                    credentials["result_size_bytes"] = len(
                        canonical_json(result).encode()
                    )
                    credentials["finish_event_id"] = event_id
                    common["result_digest"] = credentials["result_digest"]
                    if result.get("final_url_digest") is not None:
                        common["final_url_digest"] = result["final_url_digest"]
                    terminal["result_size_bytes"] = credentials["result_size_bytes"]
                except (TypeError, ValueError):
                    logger.warning("FlowPilot result canonicalization failed")
        self._post_tool_event(
            {**common, **terminal, "event_id": event_id, "sequence": 2}
        )
        self._complete_leader_binding(
            action,
            identity=identity,
            events=events,
            error=error,
            cancelled=cancelled,
        )

    def tool_cancel(self, action: ActionEvent) -> None:
        with self._lock:
            active = self._active_tools.pop(action.id, None)
            cancellation = self._reuse_cancellations.get(action.id)
            waiting = self._waiting_reuses.pop(action.id, None)
        if cancellation is not None:
            cancellation.set()
        if waiting is not None:
            self._cancel_follower(*waiting)
        if active is None:
            return
        identity, attempt, _started = active
        common = self._tool_common(identity, action, attempt)
        with self._lock:
            credentials = self._leader_credentials.get(action.id)
        if credentials is not None:
            common.update(
                {
                    key: credentials[key]
                    for key in (
                        "binding_id",
                        "input_digest",
                        "adapter_id",
                        "adapter_version",
                        "result_schema_version",
                        "executor_kind",
                    )
                }
            )
            common["reuse_receipt_version"] = "flowpilot-execution-v1"
        self._post_tool_event(
            {
                **common,
                "event_id": str(uuid.uuid4()),
                "sequence": 2,
                "event_kind": "cancel",
                "error_class": "CancelledError",
                "observed_at": datetime.now(UTC).isoformat(),
            }
        )
        self._complete_leader_binding(
            action,
            identity=identity,
            events=None,
            error=RuntimeError("CancelledError"),
            cancelled=True,
        )

    def _tool_common(
        self,
        identity: FlowPilotRequestIdentity,
        action: ActionEvent,
        attempt: int,
    ) -> dict[str, Any]:
        common: dict[str, Any] = {
            "protocol_version": "flowpilot-phase0-v2",
            "job_id": identity.job_id,
            "line_id": identity.line_id,
            "context_epoch": identity.context_epoch,
            "tail_request_id": identity.tail_request_id,
            "request_id": identity.request_id,
            "llm_call_id": identity.llm_call_id,
            "attempt": identity.attempt,
            "conversation_id": identity.conversation_id,
            "action_id": action.id,
            "tool_call_id": action.tool_call_id,
            "tool_name": action.tool_name,
            "tool_class": (
                "web"
                if action.tool_name == "web_search"
                or action.tool_name in self.config.reusable_web_tools
                else "non_web"
            ),
            "execution_attempt": attempt,
        }
        return common

    def _post_tool_event(self, payload: dict[str, Any]) -> None:
        payload.setdefault("observed_at", datetime.now(UTC).isoformat())
        request = urllib.request.Request(
            f"{self.config.control_base_url}/flowpilot/v1/events/tools",
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={
                "content-type": "application/json",
                "x-flowpilot-api-key": self.config.api_key,
            },
            method="POST",
        )
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout):
                    return
            except Exception:
                if attempt == 1:
                    logger.warning("FlowPilot tool telemetry failed", exc_info=True)

    def _post_control(self, path: str, payload: dict[str, Any]) -> None:
        payload["protocol_version"] = "flowpilot-phase0-v2"
        request = urllib.request.Request(
            f"{self.config.control_base_url}{path}",
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={
                "content-type": "application/json",
                "x-flowpilot-api-key": self.config.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout):
                pass
        except Exception as exc:
            raise RuntimeError(f"FlowPilot registration failed for {path}") from exc

    def _complete_leader_binding(
        self,
        action: ActionEvent,
        *,
        identity: FlowPilotRequestIdentity,
        events: list[Event] | None,
        error: BaseException | None,
        cancelled: bool,
    ) -> None:
        with self._lock:
            active = self._leader_bindings.pop(action.id, None)
        if active is None or active[0] != identity:
            return
        binding_id = active[1]
        reuse_identity = self._reuse_identity(identity, action)
        try:
            if error is not None or cancelled or not events:
                with self._lock:
                    self._leader_credentials.pop(action.id, None)
                self._request_json(
                    f"/flowpilot/v1/reuse/bindings/{binding_id}/fail",
                    {
                        "protocol_version": self.config.reuse_protocol_version,
                        "binding_id": binding_id,
                        "identity": reuse_identity,
                        "error_class": (
                            "CancelledError" if cancelled else type(error).__name__
                        ),
                    },
                    method="POST",
                )
                return
            if not isinstance(events[0], ObservationEvent):
                raise ValueError("leader did not produce an ObservationEvent")
            credentials = self._leader_credentials.pop(action.id)
            self._pending_publications[action.id] = {
                "protocol_version": self.config.reuse_protocol_version,
                "binding_id": binding_id,
                "identity": reuse_identity,
                "cacheable": True,
                **{
                    key: credentials[key]
                    for key in (
                        "start_event_id",
                        "finish_event_id",
                        "execution_attempt",
                        "input_digest",
                        "result_digest",
                        "result_size_bytes",
                        "result_schema_version",
                        "result",
                    )
                },
            }
        except Exception:
            logger.warning("FlowPilot leader result publication failed", exc_info=True)

    def observation_committed(self, event: ObservationEvent) -> None:
        """Called only after the authoritative history accepted this Observation."""
        with self._lock:
            publication = self._pending_publications.get(event.action_id)
        if publication is None:
            return
        if (
            payload_digest(event.observation.model_dump(mode="json"))
            != publication["result_digest"]
        ):
            logger.warning("FlowPilot committed Observation differs from FINISH")
            return
        # Retry the identical fingerprint after response loss. Never re-execute
        # the Tool or alter the already committed Observation on failure.
        for attempt in range(2):
            try:
                self._request_json(
                    f"/flowpilot/v1/reuse/bindings/{publication['binding_id']}/result",
                    publication,
                    method="POST",
                )
            except Exception:
                if attempt == 1:
                    logger.warning(
                        "FlowPilot publication failed after local commit", exc_info=True
                    )
            else:
                break
        with self._lock:
            self._pending_publications.pop(event.action_id, None)

    def _wait_for_reuse(
        self,
        binding_id: str,
        identity: dict[str, str],
        *,
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.reuse_wait_timeout
        query = urllib.parse.urlencode(identity)
        while time.monotonic() < deadline:
            if cancel_event.is_set():
                return {"decision": "cancelled"}
            decision = self._request_json(
                f"/flowpilot/v1/reuse/bindings/{binding_id}?{query}",
                None,
                method="GET",
            )
            if cancel_event.is_set():
                return {"decision": "cancelled"}
            if decision.get("decision") not in {
                "wait_and_sync_reused_result",
                "defer_wait_for_inflight",
            }:
                return decision
            cancel_event.wait(self.config.reuse_poll_interval)
        if cancel_event.is_set():
            return {"decision": "cancelled"}
        self._cancel_follower(binding_id, identity)
        return {"decision": "execute_locally"}

    def _cancel_follower(self, binding_id: str, identity: dict[str, str]) -> None:
        try:
            self._request_json(
                f"/flowpilot/v1/reuse/bindings/{binding_id}/cancel",
                {
                    "protocol_version": self.config.reuse_protocol_version,
                    "binding_id": binding_id,
                    "identity": identity,
                },
                method="POST",
            )
        except Exception:
            logger.warning("FlowPilot follower cancellation failed", exc_info=True)

    @staticmethod
    def _reuse_identity(
        identity: FlowPilotRequestIdentity, action: ActionEvent
    ) -> dict[str, str]:
        return {
            "job_id": identity.job_id,
            "line_id": identity.line_id,
            "tail_request_id": identity.tail_request_id,
            "llm_call_id": identity.llm_call_id,
            "action_id": action.id,
            "tool_call_id": action.tool_call_id,
        }

    def _request_json(
        self, path: str, payload: dict[str, Any] | None, *, method: str
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.config.control_base_url}{path}",
            data=(
                json.dumps(payload, separators=(",", ":")).encode()
                if payload is not None
                else None
            ),
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "x-flowpilot-api-key": self.config.api_key,
            },
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            value = json.loads(response.read())
        if not isinstance(value, dict):
            raise ValueError("FlowPilot returned a non-object response")
        return value

    def _fetch_authoritative_tail(self) -> dict[str, Any]:
        tail = self._find_authoritative_tail()
        if tail is None:
            raise RuntimeError(
                "FlowPilot frontier does not contain the configured line"
            )
        return tail

    def _find_authoritative_tail(self) -> dict[str, Any] | None:
        request = urllib.request.Request(
            (
                f"{self.config.control_base_url}/flowpilot/v1/jobs/"
                f"{urllib.parse.quote(self.config.job_id, safe='')}/frontier"
            ),
            headers={"x-flowpilot-api-key": self.config.api_key},
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout
            ) as response:
                payload = json.loads(response.read())
        except Exception as exc:
            raise RuntimeError("FlowPilot frontier reconciliation failed") from exc
        lines = payload.get("lines") if isinstance(payload, dict) else None
        if not isinstance(lines, list):
            raise RuntimeError("FlowPilot frontier response is malformed")
        tail = next(
            (
                item
                for item in lines
                if isinstance(item, dict) and item.get("line_id") == self.config.line_id
            ),
            None,
        )
        return tail


def _tail_phase(tail: dict[str, Any]) -> str | None:
    phase = tail.get("phase")
    if isinstance(phase, str):
        return phase
    state = tail.get("state")
    if not isinstance(state, str):
        return None
    return {
        "LLM_RUNNING": "ACTIVE",
        "NEXT_READY": "READY",
        "FINISHED": "TERMINAL",
    }.get(state, state)


def context_digest(events: list[Any]) -> str:
    payload = []
    for event in events:
        serialized = event.model_dump(mode="json", exclude_none=True)
        payload.append({"type": event.__class__.__name__, "event": serialized})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _redact_provider_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep provider ordering/identities while excluding message/tool bodies."""
    redacted: dict[str, Any] = {}
    for key, value in message.items():
        if key in {"content", "input", "output", "arguments", "result"}:
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            redacted[key] = {
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "bytes": len(encoded),
            }
        elif key == "tool_calls" and isinstance(value, list):
            calls: list[dict[str, Any]] = []
            for call in value:
                if not isinstance(call, dict):
                    calls.append({"sha256": _stable_digest(call)})
                    continue
                item = {k: v for k, v in call.items() if k != "function"}
                function = call.get("function")
                if isinstance(function, dict):
                    item["function"] = {
                        k: (
                            {"sha256": _stable_digest(v), "bytes": len(str(v))}
                            if k == "arguments"
                            else v
                        )
                        for k, v in function.items()
                    }
                calls.append(item)
            redacted[key] = calls
        else:
            redacted[key] = value
    return redacted


def _execution_tool_arguments(action: ActionEvent) -> dict[str, Any]:
    if action.action is None:
        raise ValueError("Tool Action is missing")
    if isinstance(action.action, MCPToolAction):
        return action.action.to_mcp_arguments()
    value = action.action.model_dump(mode="json", exclude={"kind", "security_risk"})
    return value


def _provider_tool_arguments(action: ActionEvent) -> dict[str, Any]:
    try:
        arguments = json.loads(action.tool_call.arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("FlowPilot Tool arguments are not valid JSON") from exc
    if not isinstance(arguments, dict):
        raise ValueError("FlowPilot Tool arguments must be a JSON object")
    return arguments


def _require_string(value: dict[str, Any], field_name: str) -> str:
    item = value.get(field_name)
    if not isinstance(item, str) or not item:
        raise ValueError(f"FlowPilot response field {field_name} is malformed")
    return item


def _event_result_size(events: list[Event]) -> int:
    observations = [
        observation.model_dump(mode="json")
        for event in events
        if isinstance((observation := getattr(event, "observation", None)), Observation)
    ]
    if not observations:
        return 0
    payload: Any = observations[0] if len(observations) == 1 else observations
    return len(canonical_json(payload).encode())


def _with_reuse_provenance(
    observation: Observation,
    _result: dict[str, Any],
    provenance: dict[str, Any],
) -> Observation:
    allowed = {
        key: provenance[key]
        for key in ("reuse_type", "observed_at", "result_schema_version")
    }
    suffix = TextContent(
        text="\n[FlowPilot reuse provenance: " + canonical_json(allowed) + "]"
    )
    return observation.model_copy(update={"content": [*observation.content, suffix]})


def _with_provider_content(
    observation: Observation, provider_content: str
) -> Observation:
    from openhands.sdk.llm import TextContent

    try:
        return observation.model_copy(
            update={"content": [TextContent(text=provider_content)]}
        )
    except Exception:
        return FlowPilotReusedObservation(
            content=[TextContent(text=provider_content)],
            is_error=observation.is_error,
        )


def _reuse_provider_content(result: dict[str, Any], provenance: dict[str, Any]) -> str:
    allowed = {
        key: provenance[key]
        for key in ("reuse_type", "observed_at", "result_schema_version")
        if key in provenance
    }
    return (
        f"{json.dumps(result, sort_keys=True, separators=(',', ':'))}\n"
        "[FlowPilot reuse provenance: "
        f"{json.dumps(allowed, sort_keys=True, separators=(',', ':'))}]"
    )
