import contextlib
import hashlib
import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path


def json_value(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value) + "\n")
    temp.replace(path)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, max_tools, max_requests, timeout):
        self.max_tools, self.max_requests = max_tools, max_requests
        self.deadline = time.monotonic() + timeout
        self.tools = self.requests = 0
        self.reason = None

    def check(self):
        if time.monotonic() >= self.deadline:
            self.reason = "task_timeout"
        if self.reason:
            raise BudgetExceeded(self.reason)

    def tool(self):
        self.check()
        if self.tools >= self.max_tools:
            self.reason = "max_tool_calls"
            raise BudgetExceeded(self.reason)
        self.tools += 1

    def request(self):
        self.check()
        if self.requests >= self.max_requests:
            self.reason = "max_llm_requests"
            raise BudgetExceeded(self.reason)
        self.requests += 1

    def snapshot(self):
        return {
            "max_tools": self.max_tools,
            "max_requests": self.max_requests,
            "tools_used": self.tools,
            "requests_used": self.requests,
            "remaining_seconds": self.remaining(),
        }

    def remaining(self):
        return max(0.01, self.deadline - time.monotonic())


class TraceRecorder:
    def __init__(self, directory, identity, load_monitor=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.blobs = self.directory / "blobs"
        self.blobs.mkdir(exist_ok=True)
        self.identity = identity.copy()
        self.load_monitor = load_monitor
        self.tool_execution_profile = {}
        self._file = (self.directory / "events.jsonl").open("a", encoding="utf-8")
        self._lock = threading.RLock()
        self.seq = 0
        self.response_requests = {}
        self.pending = defaultdict(deque)
        self.final_text = ""
        self.error_codes = []
        self.executed_counts = defaultdict(int)
        self.retrieved_docids = set()
        self.environment_tool_names = set()

    def emit(self, event, **data):
        with self._lock:
            self.seq += 1
            record = dict(
                schema_version=1,
                **self.identity,
                seq=self.seq,
                wall_time=datetime.now(UTC).isoformat(),
                monotonic_ns=time.monotonic_ns(),
                event=event,
                **data,
            )
            self._file.write(json.dumps(record, ensure_ascii=False, default=json_value) + "\n")
            self._file.flush()

    def blob(self, data):
        encoded = json.dumps(data, ensure_ascii=False, default=json_value).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = self.blobs / (digest + ".json")
        if not path.exists():
            path.write_bytes(encoded)
        return {"path": "blobs/" + path.name, "sha256": digest}

    def activity(self, phase):
        return (
            self.load_monitor.activity(phase)
            if self.load_monitor is not None
            else contextlib.nullcontext()
        )

    def t0_features(self):
        sampled = time.monotonic_ns()
        try:
            loads = os.getloadavg()
        except OSError:
            loads = (None, None, None)
        available = None
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
        except (OSError, ValueError):
            pass
        try:
            affinity = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            affinity = None
        return {
            "schema_version": 1,
            "client_load": self.load_monitor.snapshot() if self.load_monitor is not None else None,
            "host": {
                "sampled_monotonic_ns": sampled,
                "load_avg_1m": loads[0],
                "load_avg_5m": loads[1],
                "load_avg_15m": loads[2],
                "cpu_count": os.cpu_count(),
                "affinity_cpu_count": affinity,
                "mem_available_bytes": available,
            },
        }

    def request(self, request_id, payload, **metadata):
        # Only transport-level credentials are omitted; tool argument schemas are data.
        private_keys = {
            "api_key",
            "api_token",
            "extra_headers",
            "headers",
            "http_client",
            "client",
            "success_callback",
            "failure_callback",
            "callbacks",
        }
        public = {k: v for k, v in payload.items() if k not in private_keys}
        self.emit(
            "llm_request_prepared",
            request_id=request_id,
            request=self.blob(public),
            t0_features=self.t0_features(),
            tool_execution_profile=self.tool_execution_profile,
            **metadata,
        )

    def response_decision(self, request_id, response):
        """Observed actor choices are labels, never features available at request time."""
        choices = response.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message")
        finish_reason = choice.get("finish_reason")
        complete = isinstance(message, dict) and finish_reason in {"stop", "tool_calls"}
        calls = (message.get("tool_calls") or []) if isinstance(message, dict) else []
        names = [(call.get("function") or {}).get("name") for call in calls]
        unknown = any(
            name not in self.environment_tool_names | {"finish", "think"} for name in names
        )
        has_environment = any(name in self.environment_tool_names for name in names)
        first = names[0] if names else None
        kind = (
            "unavailable"
            if not complete
            else "no_tool"
            if not names
            else "environment_tool"
            if first in self.environment_tool_names
            else first
            if first in {"finish", "think"}
            else "unknown_tool"
        )
        self.emit(
            "response_action_summary",
            request_id=request_id,
            llm_response_id=response.get("id"),
            label_stage="after_response",
            response_complete=complete,
            finish_reason=finish_reason,
            has_tool_calls=bool(calls) if complete else None,
            has_environment_tool_call=(True if has_environment else None if unknown else False)
            if complete
            else None,
            has_unknown_tool_call=unknown,
            next_action_kind=kind,
            next_tool_name=first,
            tool_calls=calls,
        )

    def sdk_event(self, event):
        data = event.model_dump(mode="json")
        kind = type(event).__name__
        self.emit("sdk_event", sdk_event_type=kind, sdk_event=data)
        if kind == "ActionEvent":
            tool = event.tool_name
            response_id = str(event.llm_response_id)
            link = dict(
                tool_call_id=event.tool_call_id,
                action_event_id=str(event.id),
                request_id=self.response_requests.get(response_id),
                llm_response_id=response_id,
            )
            if event.action is not None:
                self.pending[tool].append({**link, "submitted_monotonic_ns": time.monotonic_ns()})
            else:
                self.emit("tool_not_executed", tool_name=tool, reason="invalid_action", **link)
            self.emit("tool_proposed", tool_name=tool, action=data, **link)
            if tool == "finish" and event.action is not None:
                self.final_text = event.action.model_dump().get("message", "")
        elif kind == "MessageEvent" and event.source == "agent":
            text = "\n".join(c.text for c in event.llm_message.content if hasattr(c, "text"))
            if text:
                self.final_text = text
        elif kind == "ObservationEvent":
            for name, queue in self.pending.items():
                self.pending[name] = deque(
                    link for link in queue if link["tool_call_id"] != event.tool_call_id
                )
        elif kind in ("ConversationErrorEvent", "AgentErrorEvent"):
            self.error_codes.append(data.get("code", kind))

    def start_tool(self, name, arguments, *, effective_arguments=None):
        link = (
            self.pending[name].popleft()
            if self.pending[name]
            else {"tool_call_id": None, "request_id": None}
        )
        self.executed_counts[name] += 1
        submitted = link.get("submitted_monotonic_ns")
        self.emit(
            "tool_start",
            tool_name=name,
            arguments=arguments,
            effective_arguments=effective_arguments,
            proposal_to_executor_ms=(time.monotonic_ns() - submitted) / 1e6
            if submitted is not None
            else None,
            queue_duration_ms=None,
            **link,
        )
        return link

    def close(self):
        with self._lock:
            self._file.close()
