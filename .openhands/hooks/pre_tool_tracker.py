#!/usr/bin/env python3
"""PreToolUse hook: persist start metadata for a tool call."""

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


def main() -> int:
    try:
        event = json.load(sys.stdin)
        trace_dir = Path("/home/liyachen/workspace/experiments")
        record_dir = trace_dir / "traces" / "tool-records"
        pending_dir = record_dir / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        key = _tool_key(event)
        pending_path = pending_dir / f"{key}.json"
        tmp_path = pending_path.with_suffix(".tmp")

        record = {
            "schema_version": 1,
            "record_key": key,
            "session_id": event.get("session_id"),
            "tool_call_id": event.get("tool_call_id"),
            "tool_name": event.get("tool_name"),
            "started_at": _now_iso(),
            "started_at_epoch_ns": time.time_ns(),
            "started_at_perf_ns": time.perf_counter_ns(),
            "tool_input": event.get("tool_input"),
        }

        tmp_path.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(pending_path)
        _allow()
        return 0
    except Exception as exc:
        print(f"pre_tool hook failed open: {exc}", file=sys.stderr)
        _allow()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
