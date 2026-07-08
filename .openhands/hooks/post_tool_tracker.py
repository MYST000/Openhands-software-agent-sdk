#!/usr/bin/env python3
"""PostToolUse hook: complete a tool-call timing record."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _allow() -> None:
    print(json.dumps({"decision": "allow"}))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tool_key(event: dict[str, Any]) -> str:
    payload = {
        "session_id": event.get("session_id"),
        "tool_call_id": event.get("tool_call_id"),
        "tool_name": event.get("tool_name"),
    }
    if not event.get("tool_call_id"):
        payload["tool_input"] = event.get("tool_input")
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _duration_ms(
    start: dict[str, Any], ended_perf_ns: int, ended_epoch_ns: int
) -> float:
    started_perf_ns = start.get("started_at_perf_ns")
    if isinstance(started_perf_ns, int):
        return round((ended_perf_ns - started_perf_ns) / 1_000_000, 3)

    started_epoch_ns = start.get("started_at_epoch_ns")
    if isinstance(started_epoch_ns, int):
        return round((ended_epoch_ns - started_epoch_ns) / 1_000_000, 3)

    return 0.0


def main() -> int:
    try:
        event = json.load(sys.stdin)
        trace_dir = Path("/home/liyachen/workspace/experiments")
        record_dir = trace_dir / "traces" / "tool-records"
        pending_dir = record_dir / "pending"
        complete_path = record_dir / "tool_calls.jsonl"
        record_dir.mkdir(parents=True, exist_ok=True)

        key = _tool_key(event)
        pending_path = pending_dir / f"{key}.json"

        start_record: dict[str, Any] = {}
        missing_start_record = True
        if pending_path.exists():
            start_record = json.loads(pending_path.read_text(encoding="utf-8"))
            missing_start_record = False
            pending_path.unlink(missing_ok=True)

        ended_epoch_ns = time.time_ns()
        ended_perf_ns = time.perf_counter_ns()

        record = {
            "schema_version": 1,
            "record_key": key,
            "session_id": event.get("session_id") or start_record.get("session_id"),
            "tool_call_id": event.get("tool_call_id")
            or start_record.get("tool_call_id"),
            "tool_name": event.get("tool_name") or start_record.get("tool_name"),
            "started_at": start_record.get("started_at"),
            "ended_at": _now_iso(),
            "duration_ms": None
            if missing_start_record
            else _duration_ms(start_record, ended_perf_ns, ended_epoch_ns),
            "missing_start_record": missing_start_record,
            "tool_input": event.get("tool_input") or start_record.get("tool_input"),
            "tool_response": event.get("tool_response"),
        }

        with complete_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        _allow()
        return 0
    except Exception as exc:
        print(f"post_tool hook failed open: {exc}", file=sys.stderr)
        _allow()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
