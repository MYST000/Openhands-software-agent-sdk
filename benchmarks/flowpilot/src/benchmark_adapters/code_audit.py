"""Audit an actual completed pilot; never infer benchmark correctness from a trace."""

import hashlib
import json
from collections import Counter
from pathlib import Path


def audit_attempt(attempt_dir):
    ATTEMPT = Path(attempt_dir)
    events = [json.loads(line) for line in (ATTEMPT / "events.jsonl").read_text().splitlines()]
    result = json.loads((ATTEMPT / "result.json").read_text())
    requests = {e["request_id"]: e for e in events if e["event"] == "llm_request_prepared"}
    responses = {e["request_id"]: e for e in events if e["event"] == "llm_response"}
    decisions = {e["request_id"]: e for e in events if e["event"] == "response_action_summary"}
    starts = {e["tool_call_id"]: e for e in events if e["event"] == "tool_start"}
    ends = {e["tool_call_id"]: e for e in events if e["event"] == "tool_end"}
    problems = []
    observation_matches = 0
    terminal_observations = 0
    sdk_error_header_observations = 0
    sdk_observations = {
        e["sdk_event"]["tool_call_id"]: e["sdk_event"]
        for e in events
        if e.get("sdk_event_type") == "ObservationEvent"
    }

    def blob(ref):
        raw = (ATTEMPT / ref["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
        return json.loads(raw)

    snapshots = {rid: blob(e["request"]) for rid, e in requests.items()}
    sample_rows = []
    censored_requests = []
    timeout_ids = {
        e.get("request_id")
        for e in events
        if e["event"] == "llm_error" and e.get("error_type") in {"Timeout", "APITimeoutError"}
    }
    for rid, req in requests.items():
        if rid not in responses or rid not in decisions:
            if (
                result["execution_status"] == "budget_exhausted"
                and result.get("termination_reason") == "task_timeout"
            ) or (rid in timeout_ids and result["execution_status"] == "llm_error"):
                censored_requests.append(rid)
            else:
                problems.append(f"Missing response/decision for {rid}")
            continue
        resp = blob(responses[rid]["response"])
        decision = decisions[rid]
        actions = []
        for call in decision["tool_calls"]:
            call_id = call["id"]
            end = ends.get(call_id)
            actions.append(
                dict(
                    tool_call_id=call_id,
                    tool_name=call["function"]["name"],
                    arguments_raw=call["function"]["arguments"],
                    executed=call_id in starts,
                    executor_duration_ms=end["executor_duration_ms"] if end else None,
                    round_trip_ms=end["round_trip_ms"] if end else None,
                    clock_domain=end["outcome"]["clock_domain"] if end else None,
                )
            )
        sample_rows.append(
            dict(
                task_id=result["task_id"],
                conversation_id=result["conversation_id"],
                environment_id=result["environment_id"],
                request_id=rid,
                backend="local-process-pilot",
                features=dict(
                    request_snapshot=req["request"],
                    available_at=req["wall_time"],
                    snapshot_stage=req["snapshot_stage"],
                ),
                labels=dict(
                    next_action_kind=decision["next_action_kind"],
                    has_environment_tool_call=decision["has_environment_tool_call"],
                    next_tool_name=decision["next_tool_name"],
                    actions=actions,
                    label_stage="after_response_and_execution",
                ),
                response_usage=resp.get("usage"),
            )
        )

    for call_id, start in starts.items():
        end = ends.get(call_id)
        rid = start["request_id"]
        if not end or rid not in requests or rid not in responses:
            problems.append(f"Incomplete tool linkage {call_id}")
            continue
        for key in (
            "request_id",
            "llm_response_id",
            "action_event_id",
            "environment_id",
            "conversation_id",
        ):
            if start.get(key) is None or start.get(key) != end.get(key):
                problems.append(f"Bad tool {key}: {call_id}")
        if not requests[rid]["seq"] < responses[rid]["seq"] < start["seq"] < end["seq"]:
            problems.append(f"Bad sequence for {call_id}")
        if end["executor_duration_ms"] < 0 or end["round_trip_ms"] < end["executor_duration_ms"]:
            problems.append(f"Bad duration for {call_id}")
        sdk = sdk_observations.get(call_id, {})
        expected_text = end["model_observation"]
        if sdk.get("observation", {}).get("is_error"):
            from openhands.sdk.tool.schema import Observation

            expected_text = Observation.ERROR_MESSAGE_HEADER + expected_text
            sdk_error_header_observations += 1
        found = False
        for later_id, later in requests.items():
            if later["seq"] <= end["seq"]:
                continue
            for msg in snapshots[later_id].get("messages", []):
                if msg.get("role") == "tool" and msg.get("tool_call_id") == call_id:
                    content = msg.get("content")
                    text = (
                        "".join(c.get("text", "") for c in content)
                        if isinstance(content, list)
                        else content
                    )
                    if text == expected_text:
                        found = True
            if found:
                break
        observation_matches += int(found)
        if not found:
            no_next_request = not any(req["seq"] > end["seq"] for req in requests.values())
            sdk = sdk_observations.get(call_id, {})
            observed_text = "".join(
                c.get("text", "") for c in sdk.get("observation", {}).get("content", [])
            )
            if no_next_request and observed_text == end["model_observation"]:
                terminal_observations += 1
            else:
                problems.append(f"Observation missing from subsequent request: {call_id}")
    unexpected_errors = [
        e
        for e in events
        if e["event"] in ("tool_error", "llm_error", "task_error")
        and e.get("error_type") != "BudgetExceeded"
        and not (e["event"] == "llm_error" and e.get("request_id") in censored_requests)
        and not (
            e["event"] == "task_error"
            and e.get("reason") == "task_timeout"
            and result.get("termination_reason") == "task_timeout"
            and result["execution_status"] == "budget_exhausted"
        )
        and not (
            e["event"] == "task_error"
            and e.get("error_type") in {"Timeout", "APITimeoutError"}
            and censored_requests
            and result["execution_status"] == "llm_error"
        )
        and not (
            e["event"] == "task_error"
            and e.get("error_type") == "ConversationRunError"
            and "LLMTimeoutError" in result.get("errors", [])
            and censored_requests
            and result["execution_status"] == "llm_error"
        )
    ]
    if unexpected_errors:
        problems.append("Unexpected execution error events present")
    usage = [row["response_usage"] for row in sample_rows]
    report = dict(
        backend="local-process-pilot",
        task_id=result["task_id"],
        request_count=len(requests),
        response_count=len(responses),
        budget_censored_request_ids=censored_requests,
        trace_status="complete" if not censored_requests else "partial-request-timeout",
        tool_count=len(starts),
        matched_tool_ends=len(ends),
        observation_in_subsequent_request_count=observation_matches,
        terminal_observations_recorded_without_next_request=terminal_observations,
        observations_with_pinned_sdk_error_header=sdk_error_header_observations,
        termination_reason=result["termination_reason"],
        proposed_action_counts=dict(
            Counter(a["tool_name"] for row in sample_rows for a in row["labels"]["actions"])
        ),
        executed_tool_counts=dict(Counter(e["tool_name"] for e in starts.values())),
        prompt_tokens_sum=sum(u["prompt_tokens"] for u in usage),
        max_prompt_tokens=max((u["prompt_tokens"] for u in usage), default=0),
        completion_tokens_sum=sum(u["completion_tokens"] for u in usage),
        solution_bytes=(ATTEMPT / "artifacts/solution.py").stat().st_size,
        execution_status=result["execution_status"],
        problems=problems,
        audit_passed=not problems,
        evaluation_note="separate benchmark evaluation artifact",
        feature_note="Only request_snapshot is predictor input; response_usage and labels are observed afterward.",
    )
    (ATTEMPT / "prediction_samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample_rows)
    )
    (ATTEMPT / "trace_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    import sys

    print(json.dumps(audit_attempt(sys.argv[1]), indent=2))
