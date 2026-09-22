"""Materialize causal request inputs and observed labels without modifying raw traces."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .tracing import write_json

CONTROL_TOOLS = {"think", "finish"}
IDENTITY_FIELDS = (
    "run_id",
    "task_id",
    "dataset_id",
    "dataset_revision",
    "split",
    "attempt_id",
    "conversation_id",
    "environment_id",
    "episode_id",
    "replica_id",
    "worker_slot",
    "host_id",
    "clock_domain",
    "task_group_id",
    "research_split",
)


class TraceIntegrityError(ValueError):
    pass


def _reject_json_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _finite_json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite JSON number: {value}")
    return result


def _strict_json_loads(value):
    return json.loads(value, parse_constant=_reject_json_constant, parse_float=_finite_json_float)


def checked_blob(attempt, reference):
    root = Path(attempt).resolve()
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root / "blobs"):
        raise TraceIntegrityError("Blob path escapes attempt blobs")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise TraceIntegrityError(f"Blob hash mismatch: {reference['path']}")
    try:
        return _strict_json_loads(raw)
    except ValueError as exc:
        raise TraceIntegrityError(f"Invalid JSON blob: {reference['path']}") from exc


def _unique(events, event_name, key):
    result = {}
    for event in events:
        if event["event"] != event_name:
            continue
        identity = event.get(key)
        if not identity or identity in result:
            raise TraceIntegrityError(f"Missing/duplicate {event_name}.{key}: {identity}")
        result[identity] = event
    return result


def _same_identity(left, right):
    for key in (*IDENTITY_FIELDS, "request_id", "llm_response_id", "action_event_id"):
        if key in left and key in right and left[key] != right[key]:
            raise TraceIntegrityError(f"Mismatched {key}")


def _read_trace(attempt):
    try:
        events = [
            _strict_json_loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()
        ]
    except (ValueError, OSError) as exc:
        raise TraceIntegrityError("Unreadable or truncated events.jsonl") from exc
    previous_seq, previous_time = 0, -1
    identity = {}
    for event in events:
        if (
            not isinstance(event, dict)
            or type(event.get("seq")) is not int
            or type(event.get("monotonic_ns")) is not int
        ):
            raise TraceIntegrityError("Invalid trace sequence/clock")
        if event["seq"] <= previous_seq or event["monotonic_ns"] < previous_time:
            raise TraceIntegrityError("Non-monotonic trace sequence/clock")
        previous_seq, previous_time = event["seq"], event["monotonic_ns"]
        # Environment/conversation IDs appear after preparation; once known they are stable.
        for key in IDENTITY_FIELDS:
            value = event.get(key)
            if value is None:
                continue
            if key in identity and identity[key] != value:
                raise TraceIntegrityError(f"Mismatched trace {key}")
            identity[key] = value
    indexes = {
        name: _unique(events, name, key)
        for name, key in (
            ("llm_request_prepared", "request_id"),
            ("llm_response", "request_id"),
            ("response_action_summary", "request_id"),
            ("tool_start", "tool_call_id"),
            ("tool_end", "tool_call_id"),
            ("tool_error", "tool_call_id"),
        )
    }
    # Old budget-denial events without call IDs remain unresolved, not fabricated links.
    indexes["tool_not_executed"] = _unique(
        [event for event in events if event.get("tool_call_id")],
        "tool_not_executed",
        "tool_call_id",
    )
    return events, indexes


def _check_links(attempt, events, indexes):
    requests = indexes["llm_request_prepared"]
    responses = indexes["llm_response"]
    decisions = indexes["response_action_summary"]
    starts, ends = indexes["tool_start"], indexes["tool_end"]
    proposals = {}
    for rid, response in responses.items():
        if rid not in requests or response["seq"] <= requests[rid]["seq"]:
            raise TraceIntegrityError("Orphan/out-of-order response")
        _same_identity(requests[rid], response)
        duration = response.get("duration_ms")
        if duration is not None and (
            type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0
        ):
            raise TraceIntegrityError("Invalid LLM transport duration")
        canonical = checked_blob(attempt, response["response"])
        if not isinstance(canonical, dict) or not canonical.get("id"):
            raise TraceIntegrityError("Response blob has no identity")
        if response.get("llm_response_id") != canonical["id"]:
            raise TraceIntegrityError("Response event does not match response blob identity")
    for rid, decision in decisions.items():
        if rid not in responses or decision["seq"] <= responses[rid]["seq"]:
            raise TraceIntegrityError("Orphan/out-of-order response decision")
        _same_identity(responses[rid], decision)
        response = checked_blob(attempt, responses[rid]["response"])
        choices = response.get("choices") or []
        if not isinstance(choices, list) or (choices and not isinstance(choices[0], dict)):
            raise TraceIntegrityError("Invalid response choices")
        choice = choices[0] if choices else {}
        message = choice.get("message")
        calls = (message.get("tool_calls") or []) if isinstance(message, dict) else []
        complete = isinstance(message, dict) and choice.get("finish_reason") in {
            "stop",
            "tool_calls",
        }
        if (
            decision.get("llm_response_id") != response["id"]
            or decision.get("response_complete") is not complete
            or decision.get("finish_reason") != choice.get("finish_reason")
        ):
            raise TraceIntegrityError("Decision does not match canonical response status")
        if not isinstance(calls, list) or decision.get("tool_calls") != calls:
            raise TraceIntegrityError("Decision does not match response proposals")
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise TraceIntegrityError("Invalid response proposal")
            cid = call.get("id")
            if not cid or cid in proposals:
                raise TraceIntegrityError("Missing/duplicate proposal tool_call_id")
            proposals[cid] = (rid, call)
    for cid, start in starts.items():
        if cid not in proposals:
            raise TraceIntegrityError("Orphan tool start")
        rid, call = proposals[cid]
        if start.get("request_id") != rid or start.get("tool_name") != call["function"]["name"]:
            raise TraceIntegrityError("Tool start does not match proposal")
        _same_identity(responses[rid], start)
        if start["seq"] <= decisions[rid]["seq"]:
            raise TraceIntegrityError("Tool started before its proposal")
    for name in ("tool_end", "tool_error"):
        for cid, end in indexes[name].items():
            if cid not in starts or end["seq"] <= starts[cid]["seq"]:
                raise TraceIntegrityError("Orphan/out-of-order tool end/error")
            _same_identity(starts[cid], end)
            if end.get("tool_name") != starts[cid].get("tool_name"):
                raise TraceIntegrityError("Mismatched tool name")
            durations = [end.get("round_trip_ms")]
            if name == "tool_end":
                durations.append(end.get("executor_duration_ms"))
            if any(
                x is not None and (type(x) not in (int, float) or not math.isfinite(x) or x < 0)
                for x in durations
            ):
                raise TraceIntegrityError("Invalid duration")
            if (
                name == "tool_end"
                and all(x is not None for x in durations)
                and durations[0] < durations[1]
            ):
                raise TraceIntegrityError("Executor exceeds round-trip duration")
    if indexes["tool_error"].keys() & ends.keys():
        raise TraceIntegrityError("Tool has both end and error")
    for cid, event in indexes["tool_not_executed"].items():
        if cid not in proposals or cid in starts:
            raise TraceIntegrityError("Invalid tool rejection linkage")
        rid, call = proposals[cid]
        if event.get("request_id") != rid or event.get("tool_name") != call["function"]["name"]:
            raise TraceIntegrityError("Tool rejection does not match proposal")
        _same_identity(responses[rid], event)
        if event["seq"] <= decisions[rid]["seq"]:
            raise TraceIntegrityError("Tool rejected before its proposal")


def _arguments(raw):
    try:
        value = _strict_json_loads(raw) if isinstance(raw, str) else raw
        # Legacy providers may supply an object instead of a JSON string.
        json.dumps(value, allow_nan=False)
        return (value, True) if isinstance(value, dict) else (value, False)
    except (TypeError, ValueError):
        return None, False


def _action(call, index, indexes, rejected):
    cid, name = call["id"], call["function"]["name"]
    start, end = indexes["tool_start"].get(cid), indexes["tool_end"].get(cid)
    error = indexes["tool_error"].get(cid)
    args, valid = _arguments(call["function"].get("arguments"))
    outcome = (end.get("outcome") or {}) if end else None
    status = (
        "completed"
        if end
        else "execution_error"
        if error
        else "missing_end"
        if start
        else "not_executed"
        if cid in rejected
        else "control"
        if name in CONTROL_TOOLS
        else "proposed_only"
    )
    validated = start.get("arguments") if start else None
    effective = start.get("effective_arguments") if start else None
    if effective is None and validated:
        effective = {k: v for k, v in validated.items() if k not in {"kind", "summary"}}
    return {
        "batch_index": index,
        "tool_call_id": cid,
        "tool_name": name,
        "arguments_raw": call["function"].get("arguments"),
        "arguments_parsed": args,
        "arguments_json_valid": valid,
        "validated_executor_arguments": validated,
        "effective_arguments": effective,
        "executed": start is not None,
        "execution_status": status,
        "not_executed_reason": rejected.get(cid, {}).get("reason"),
        "effective_timeout_s": start.get("effective_timeout_s", (effective or {}).get("timeout"))
        if start
        else None,
        "executor_duration_ms": end.get("executor_duration_ms") if end else None,
        "round_trip_ms": (end or error or {}).get("round_trip_ms"),
        "clock_domain": (outcome or {}).get("clock_domain")
        or (end or {}).get("executor_clock_domain"),
        "execution_outcome": {k: outcome.get(k) for k in ("exit_code", "timed_out", "truncated")}
        if outcome
        else None,
        "normal_completion_right_censored": outcome.get("timed_out") if outcome else None,
        "observed_executor_wall_time_available": end is not None
        and end.get("executor_duration_ms") is not None,
    }


_CLIENT_LOAD_FIELDS = frozenset(
    {
        "sampled_monotonic_ns",
        "active_sessions",
        "llm_inflight",
        "tool_inflight",
        "setup_active",
        "queued_sessions",
        "max_sessions",
        "scope",
    }
)
_HOST_FIELDS = frozenset(
    {
        "sampled_monotonic_ns",
        "load_avg_1m",
        "load_avg_5m",
        "load_avg_15m",
        "cpu_count",
        "affinity_cpu_count",
        "mem_available_bytes",
    }
)
_LOAD_AVERAGES = frozenset({"load_avg_1m", "load_avg_5m", "load_avg_15m"})


def _t0_snapshot(value, request_time):
    """Project only measured causal telemetry; never forward arbitrary nested data."""
    if value is None:
        return None
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        raise TraceIntegrityError("Invalid t0_features schema_version")
    if value["schema_version"] != 1:
        raise TraceIntegrityError("Unsupported t0_features schema_version")
    result: dict[str, Any] = {"schema_version": 1}
    for group, allowed in (("client_load", _CLIENT_LOAD_FIELDS), ("host", _HOST_FIELDS)):
        source = value.get(group)
        if source is None:
            result[group] = None
            continue
        if not isinstance(source, dict):
            raise TraceIntegrityError(f"Invalid t0_features.{group}")
        projected = {}
        for key in sorted(allowed):
            item = source.get(key)
            if item is not None:
                if key == "scope":
                    if item != "campaign":
                        raise TraceIntegrityError("Invalid client_load scope")
                elif key in _LOAD_AVERAGES:
                    if type(item) not in (int, float) or not math.isfinite(item) or item < 0:
                        raise TraceIntegrityError(f"Invalid t0_features.{group}.{key}")
                elif type(item) is not int or item < 0:
                    raise TraceIntegrityError(f"Invalid t0_features.{group}.{key}")
            projected[key] = item
        sampled = projected.get("sampled_monotonic_ns")
        if sampled is not None and sampled > request_time:
            raise TraceIntegrityError("Future telemetry cannot be a T0 feature")
        result[group] = projected
    return result


def _environment_features(value):
    """Keep semantic runtime capabilities; instance IDs and paths stay in raw events."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TraceIntegrityError("Invalid environment metadata")
    allowed = {
        "backend": str,
        "isolation": str,
        "official_docker_evaluation": bool,
        "retriever": str,
        "sqlite_version": str,
        "document_count": int,
    }
    result = {}
    for key, expected in allowed.items():
        if key not in value:
            continue
        item = value[key]
        if item is not None and type(item) is not expected:
            raise TraceIntegrityError(f"Invalid environment feature: {key}")
        if key == "document_count" and item is not None and item < 0:
            raise TraceIntegrityError("Invalid environment feature: document_count")
        result[key] = item
    return result


def _features(request, previous_events):
    history = []
    for event in previous_events:
        if event["event"] not in {"tool_end", "tool_error"}:
            continue
        outcome = event.get("outcome")
        history.append(
            {
                "request_id": event.get("request_id"),
                "tool_call_id": event.get("tool_call_id"),
                "tool_name": event["tool_name"],
                "execution_status": "completed"
                if event["event"] == "tool_end"
                else "execution_error",
                "executor_duration_ms": event.get("executor_duration_ms"),
                "round_trip_ms": event.get("round_trip_ms"),
                "executor_clock_domain": outcome.get("clock_domain") if outcome else None,
                "round_trip_clock_domain": "controller-process-monotonic",
                "execution_outcome": {
                    key: outcome.get(key) for key in ("exit_code", "timed_out", "truncated")
                }
                if outcome
                else None,
                "error_type": event.get("error_type"),
            }
        )
    environment = next(
        (e.get("metadata") for e in reversed(previous_events) if e["event"] == "environment_ready"),
        None,
    )
    return {
        "request_snapshot": request["request"],
        "t0_features": _t0_snapshot(request.get("t0_features"), request["monotonic_ns"]),
        "available_at": request["wall_time"],
        "monotonic_ns": request["monotonic_ns"],
        "seq": request["seq"],
        "snapshot_stage": request.get("snapshot_stage"),
        "request_role": request.get("request_role", "actor"),
        "budget_at_t0": request.get("budget_at_t0"),
        "tool_execution_profile": request.get("tool_execution_profile"),
        "environment_known_at_t0": _environment_features(environment),
        "prior_tool_executions": history,
    }


def load_t0_input(attempt_dir, row):
    """The predictor-facing loader reads only the explicit T0 feature whitelist."""
    features = row["features"]
    return {
        "snapshot": checked_blob(attempt_dir, features["request_snapshot"]),
        "t0_features": _t0_snapshot(features.get("t0_features"), features["monotonic_ns"]),
        "budget_at_t0": features.get("budget_at_t0"),
        "tool_execution_profile": features.get("tool_execution_profile"),
        "environment": _environment_features(features.get("environment_known_at_t0")),
        "prior_tool_executions": features.get("prior_tool_executions", []),
    }


def _public_task(attempt, requests):
    path = attempt / "public_task.json"
    if not path.exists():
        return {}
    try:
        task = _strict_json_loads(path.read_text())
    except (ValueError, OSError) as exc:
        raise TraceIntegrityError("Unreadable public_task.json") from exc
    if not isinstance(task, dict) or not isinstance(task.get("public_metadata", {}), dict):
        raise TraceIntegrityError("Invalid public_task metadata")
    for task_key, request_key in (
        ("task_id", "task_id"),
        ("dataset_id", "dataset_id"),
        ("revision", "dataset_revision"),
        ("split", "split"),
    ):
        if not isinstance(task.get(task_key), str) or not task[task_key]:
            raise TraceIntegrityError(f"Missing public_task identity: {task_key}")
        if any(request.get(request_key) != task[task_key] for request in requests):
            raise TraceIntegrityError(f"Mismatched public_task {task_key}")
    metadata = task.get("public_metadata", {})
    if "research_split" in metadata and metadata["research_split"] != task["split"]:
        raise TraceIntegrityError("Mismatched public_task research_split")
    for field in ("task_group_id", "research_split"):
        if field in metadata and any(
            request.get(field) is not None and request[field] != metadata[field]
            for request in requests
        ):
            raise TraceIntegrityError(f"Mismatched public_task {field}")
    return task


def build_prediction_rows(attempt_dir):
    attempt = Path(attempt_dir)
    events, indexes = _read_trace(attempt)
    _check_links(attempt, events, indexes)
    requests = list(indexes["llm_request_prepared"].values())
    task = _public_task(attempt, requests)
    metadata = task.get("public_metadata", {})
    rejected = indexes["tool_not_executed"]
    rows = []
    for i, request in enumerate(requests):
        rid = request["request_id"]
        response_event = indexes["llm_response"].get(rid)
        response = checked_blob(attempt, response_event["response"]) if response_event else None
        request_to_response = (
            (response_event["monotonic_ns"] - request["monotonic_ns"]) / 1e6
            if response_event
            else None
        )
        decision = indexes["response_action_summary"].get(rid)
        snapshot = checked_blob(attempt, request["request"])
        usage = (response or {}).get("usage") or {}
        cap = snapshot.get("max_completion_tokens", snapshot.get("max_tokens"))
        cap_reached = bool(cap and usage.get("completion_tokens", 0) >= cap)
        complete = bool(decision and decision.get("response_complete"))
        calls = decision.get("tool_calls", []) if decision else []
        actions = [_action(call, j, indexes, rejected) for j, call in enumerate(calls)]
        eligible = complete and not cap_reached
        first = actions[0] if actions else None
        environment_names = {
            t.get("function", {}).get("name") for t in snapshot.get("tools", [])
        } - CONTROL_TOOLS
        unknown = any(a["tool_name"] not in environment_names | CONTROL_TOOLS for a in actions)
        first_environment = first["tool_name"] in environment_names if first else False
        any_environment = any(a["tool_name"] in environment_names for a in actions)
        first_kind = (
            "unavailable"
            if not complete
            else "no_tool"
            if first is None
            else "environment_tool"
            if first_environment
            else first["tool_name"]
            if first["tool_name"] in CONTROL_TOOLS
            else "unknown_tool"
        )
        env_actions = [a for a in actions if a["tool_name"] in environment_names]
        all_ended = bool(env_actions) and all(
            a["execution_status"] == "completed" for a in env_actions
        )
        next_request = requests[i + 1] if i + 1 < len(requests) else None
        next_gap = (
            (next_request["monotonic_ns"] - response_event["monotonic_ns"]) / 1e6
            if next_request and response_event
            else None
        )
        batch_span = None
        if all_ended:
            batch_span = (
                max(indexes["tool_end"][a["tool_call_id"]]["monotonic_ns"] for a in env_actions)
                - min(indexes["tool_start"][a["tool_call_id"]]["monotonic_ns"] for a in env_actions)
            ) / 1e6
        rows.append(
            {
                "schema_version": 3,
                **{k: request.get(k) for k in IDENTITY_FIELDS},
                "request_id": rid,
                "logical_request_id": request.get("logical_request_id"),
                "task_group_id": metadata.get("task_group_id")
                or request.get("task_group_id")
                or f"{request.get('dataset_id')}/{request.get('task_id')}",
                "research_split": metadata.get("research_split")
                or request.get("research_split")
                or request.get("split"),
                "features": _features(request, [e for e in events if e["seq"] < request["seq"]]),
                "labels": {
                    "label_stage": "after_response_and_execution",
                    "response_available": response is not None,
                    "response_complete": complete,
                    "output_token_cap_reached": cap_reached,
                    "finish_reason": decision.get("finish_reason") if decision else None,
                    "next_action_kind": first_kind,
                    "next_tool_name": first["tool_name"] if first else None,
                    "first_is_environment": first_environment if complete else None,
                    "has_environment_tool_call": (
                        True if any_environment else None if unknown else False
                    )
                    if complete
                    else None,
                    "has_unknown_tool_call": unknown if complete else None,
                    "actions": actions,
                    "response_usage": usage if response else None,
                    "llm_transport_duration_ms": response_event.get("duration_ms")
                    if response_event
                    else None,
                    "request_to_response_ms": request_to_response,
                    "llm_duration_clock_domain": request.get("clock_domain")
                    or "controller-process-monotonic",
                    "environment_batch_span_ms": batch_span,
                    "next_request_prepared_gap_ms": next_gap,
                    "gap_clock_domain": request.get("clock_domain")
                    or "controller-process-monotonic",
                    "next_request_observed": next_request is not None,
                },
                "masks": {
                    "next_action": eligible,
                    "complete_arguments": eligible
                    and bool(actions)
                    and all(a["arguments_json_valid"] for a in actions),
                    "first_executor_duration": eligible
                    and first_environment
                    and bool(first and first["observed_executor_wall_time_available"]),
                    "first_round_trip_duration": eligible
                    and first_environment
                    and bool(first and first["round_trip_ms"] is not None),
                    "environment_batch_duration": eligible and all_ended,
                },
            }
        )
    return rows


def write_prediction_dataset(attempt_dirs, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    sources = []
    source_paths, request_keys = set(), set()
    try:
        with (
            (output / "inputs.jsonl").open("w") as inputs,
            (output / "targets.jsonl").open("w") as targets,
            (output / "samples.jsonl").open("w") as samples,
        ):
            for attempt in map(Path, attempt_dirs):
                attempt = attempt.resolve()
                if attempt in source_paths:
                    raise TraceIntegrityError(f"Duplicate source attempt: {attempt}")
                source_paths.add(attempt)
                rows = build_prediction_rows(attempt)
                task_path = attempt / "public_task.json"
                sources.append(
                    {
                        "attempt": str(attempt),
                        "events_sha256": hashlib.sha256(
                            (attempt / "events.jsonl").read_bytes()
                        ).hexdigest(),
                        "public_task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest()
                        if task_path.exists()
                        else None,
                    }
                )
                for row in rows:
                    # Physical placement must not allow duplicate logical samples.
                    request_key = tuple(
                        row[k]
                        for k in (
                            "run_id",
                            "task_id",
                            "dataset_id",
                            "dataset_revision",
                            "split",
                            "attempt_id",
                            "conversation_id",
                            "request_id",
                        )
                    )
                    if request_key in request_keys:
                        raise TraceIntegrityError(f"Duplicate request identity: {request_key}")
                    request_keys.add(request_key)
                    key = {
                        k: row[k]
                        for k in (
                            *IDENTITY_FIELDS,
                            "request_id",
                            "logical_request_id",
                            "task_group_id",
                            "research_split",
                        )
                    }
                    key["source_attempt"] = str(attempt.resolve())
                    inputs.write(
                        json.dumps({**key, "features": row["features"]}, ensure_ascii=False) + "\n"
                    )
                    targets.write(
                        json.dumps(
                            {**key, "labels": row["labels"], "masks": row["masks"]},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    samples.write(
                        json.dumps(
                            {**row, "source_attempt": str(attempt.resolve())}, ensure_ascii=False
                        )
                        + "\n"
                    )
                    counts["requests"] += 1
                    counts["eligible_next_action"] += row["masks"]["next_action"]
                    counts["missing_response"] += not row["labels"]["response_available"]
                    counts["output_cap_reached"] += row["labels"]["output_token_cap_reached"]
        files = {}
        for name in ("inputs.jsonl", "targets.jsonl", "samples.jsonl"):
            path = output / name
            files[name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        manifest = {
            "schema_version": 3,
            **dict(counts),
            "sources": sources,
            "files": files,
            "feature_policy": "Use inputs.jsonl + load_t0_input only. targets/samples contain future observations.",
            "duration_policy": "Measured wall time is observed; latent normal completion is right censored on timeout. Missing execution is null.",
        }
        write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        write_json(output / "FAILED.json", {"error_type": type(exc).__name__, "error": str(exc)})
        raise
