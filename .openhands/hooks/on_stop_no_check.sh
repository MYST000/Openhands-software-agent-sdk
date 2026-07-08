#!/usr/bin/env bash
# Lightweight Stop hook for deep-search experiments.
#
# This hook intentionally skips code-quality gates such as pre-commit, pytest,
# and CI checks. It only records that the agent attempted to finish, then allows
# OpenHands to stop.

set -u

PROJECT_DIR="${OPENHANDS_PROJECT_DIR:-$(pwd)}"
SESSION_ID="${OPENHANDS_SESSION_ID:-unknown}"
TRACE_DIR="${PROJECT_DIR}/traces/hook-records"
STOP_LOG="${TRACE_DIR}/stop_hooks.jsonl"

mkdir -p "$TRACE_DIR"

python3 - "$STOP_LOG" "$PROJECT_DIR" "$SESSION_ID" <<'PY'
import json
import sys
from datetime import datetime, timezone

log_path, project_dir, session_id = sys.argv[1:4]
record = {
    "schema_version": 1,
    "hook": "stop",
    "decision": "allow",
    "reason": "deep_search_lightweight_stop",
    "project_dir": project_dir,
    "session_id": session_id,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

with open(log_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY

echo '{"decision":"allow","reason":"deep_search_lightweight_stop"}'
exit 0
