"""Optional benchmark transport for T1 prediction and original executor RTT.

No ML dependencies in the SDK process. Reuse uses the SDK's existing protocol;
actual executor events alone feed the RTT learner.
"""

import hashlib
import json
import os
import queue
import threading
import time
from collections import Counter, OrderedDict, deque
from dataclasses import asdict, replace
from pathlib import Path

from .prediction_export import _features
from .tracing import TraceRecorder, write_json


def benchmark_flowpilot_config(config, identity):
    from openhands.sdk.flowpilot import FlowPilotConfig

    retrieval = config.dataset.kind in {"hotpot", "browsecomp"}
    scope = hashlib.sha256(
        json.dumps(
            {
                "benchmark": config.dataset.kind,
                "retrieval": asdict(config.retrieval),
                "observation": "RetrievalObservation-v1",
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    tools = (
        ("search", "get_document")
        if config.dataset.kind == "browsecomp"
        else ("search", "read_document")
    )
    adapter = FlowPilotConfig(
        enabled=True,
        gateway_url=os.environ["FLOWPILOT_PREDICTOR_GATEWAY"],
        api_key=os.environ["FLOWPILOT_INGRESS_API_KEY"],
        job_id=identity["run_id"],
        line_id=identity["attempt_id"] + "-" + identity["task_id"],
        deployment_id=os.environ.get("FLOWPILOT_REUSE_DEPLOYMENT_ID", "local"),
        namespace_id=os.environ.get("FLOWPILOT_REUSE_DEFAULT_NAMESPACE", "default"),
        exact_reuse_enabled=retrieval,
        reusable_web_tools=tools if retrieval else (),
        data_source_constraints=(f"benchmark:{config.dataset.kind}", f"corpus-config:{scope}")
        if retrieval
        else (),
        timeout=10,
    )

    if profile_path := os.environ.get("FLOWPILOT_EXPERIMENT_PROFILE"):
        profile = json.loads(Path(profile_path).read_text())
        llm = profile["openhands"]["llm"]
        extra = llm["litellm_extra_body"]
        gateway = "http://{host}:{port}".format(**profile["flowpilot"])
        expected = {k: llm[k] for k in (
            "temperature", "top_p", "max_input_tokens", "max_output_tokens", "timeout")}
        expected.update({k: extra[k] for k in (
            "top_k", "min_p", "presence_penalty", "repetition_penalty")})
        expected.update(model="openai/" + profile["vllm"]["args"]["served_model_name"],
            base_url=gateway + "/v1",
            enable_thinking=extra["chat_template_kwargs"]["enable_thinking"])
        mismatches = [name for name, value in expected.items()
                      if getattr(config.llm, name) != value]
        if config.runtime.max_iterations != profile["openhands"]["conversation"]["max_iteration_per_run"]:
            mismatches.append("max_iterations")
        if config.llm.seed not in profile["workload"]["seeds"]:
            mismatches.append("seed")
        if adapter.gateway_url.rstrip("/") != gateway:
            mismatches.append("FLOWPILOT_PREDICTOR_GATEWAY")
        if mismatches:
            raise ValueError("Benchmark differs from experiment profile: " + ", ".join(mismatches))
        options = dict(profile["openhands"]["flowpilot"])
        options["reusable_web_tools"] = tuple(
            name for name in tools if retrieval and name in options["reusable_web_tools"])
        for flag in ("exact_reuse_enabled", "semantic_reuse_enabled", "deferred_context_enabled"):
            options[flag] = bool(options[flag] and retrieval)
        adapter = replace(adapter, **options,
            deployment_id=profile["flowpilot"]["reuse_deployment_id"],
            namespace_id=profile["flowpilot"]["reuse_default_namespace"])
    return adapter


class PredictorTraceRecorder(TraceRecorder):
    def __init__(self, *args, flowpilot_config, benchmark, **kwargs):
        self.flowpilot_config = flowpilot_config
        self.benchmark = benchmark
        self._environment = None
        self._history = {}
        self._request_record = None
        self._identities = OrderedDict()
        self._feedback_queue = queue.Queue(maxsize=128)
        self.predictor_metrics = Counter()
        super().__init__(*args, tool_timing_observer=self._timing, **kwargs)
        self._feedback_thread = threading.Thread(
            target=self._feedback_worker, name="predictor-feedback", daemon=True
        )
        self._feedback_thread.start()

    def emit(self, event, **data):
        record = super().emit(event, **data)
        if event == "environment_ready":
            self._environment = record
        elif event == "llm_request_prepared":
            self._request_record = record
        elif event in {"tool_end", "tool_error"}:
            self._history.setdefault(record["tool_name"], deque(maxlen=64)).append(record)
        return record

    def request(self, request_id, payload, **metadata):
        super().request(request_id, payload, **metadata)
        headers = payload.setdefault("extra_headers", {})
        required = (
            "job_id",
            "line_id",
            "request_id",
            "tail_request_id",
            "llm_call_id",
            "context_epoch",
        )
        identity = {key: headers["x-flowpilot-" + key.replace("_", "-")] for key in required}
        identity["context_epoch"] = int(identity["context_epoch"])
        identity["attempt"] = int(headers["x-flowpilot-request-attempt"])
        identity["execution_attempt"] = 1
        self._identities[request_id] = identity
        while len(self._identities) > 4096:
            self._identities.popitem(last=False)
        history = sorted(
            [r for records in self._history.values() for r in records], key=lambda r: r["seq"]
        )
        features = _features(
            self._request_record, ([self._environment] if self._environment else []) + history
        )
        # Keep the original feature projection; omit trace-file pointers and fields
        # not consumed by this model from the bounded transport envelope.
        features.pop("request_snapshot", None)
        features["prior_tool_executions"] = [
            {k: h[k] for k in ("tool_name", "round_trip_ms", "execution_status")}
            for h in features["prior_tool_executions"]
        ]
        now = time.monotonic_ns()
        load = (features.get("t0_features") or {}).get("client_load") or {}
        context = dict(
            schema_version=1,
            benchmark=self.benchmark,
            dataset_revision=self.identity["dataset_revision"],
            replica_id=self.identity.get("replica_id"),
            environment_tools=sorted(self.environment_tool_names),
            features=features,
            snapshot_age_ms=(now - features["monotonic_ns"]) / 1e6,
            load_age_ms=(now - load["sampled_monotonic_ns"]) / 1e6
            if load.get("sampled_monotonic_ns")
            else 0,
        )
        headers["x-flowpilot-predictor-context"] = json.dumps(
            context, ensure_ascii=True, separators=(",", ":")
        )
        self.predictor_metrics["request_contexts"] += 1

    def _timing(self, timing):
        identity = self._identities.get(timing["request_id"])
        if not identity or not timing["tool_call_id"]:
            self.predictor_metrics["unbound_feedback"] += 1
            return
        payload = dict(
            identity,
            tool_call_id=timing["tool_call_id"],
            event_id=f"{identity['request_id']}:{timing['seq']}",
            round_trip_ms=timing["round_trip_ms"],
            timing_scope="client_round_trip",
            status="cancelled"
            if timing["timed_out"]
            else "execution_error"
            if timing["execution_error"]
            else "completed",
        )
        try:
            self._feedback_queue.put_nowait(payload)
        except queue.Full:
            self.predictor_metrics["feedback_queue_full"] += 1

    def _feedback_worker(self):
        import httpx

        with httpx.Client(timeout=self.flowpilot_config.timeout, trust_env=False) as client:
            while True:
                payload = self._feedback_queue.get()
                try:
                    if payload is None:
                        return
                    response = client.post(
                        self.flowpilot_config.control_base_url + "/flowpilot/v1/predictor/feedback",
                        headers={"x-flowpilot-api-key": self.flowpilot_config.api_key},
                        json=payload,
                    )
                    response.raise_for_status()
                    self.predictor_metrics["feedback_" + response.json()["status"]] += 1
                except Exception:
                    self.predictor_metrics["feedback_errors"] += 1
                finally:
                    self._feedback_queue.task_done()

    def sdk_event(self, event):
        # Reused observations bypass the executor, but still belong to benchmark
        # evidence. Do not manufacture tool_end or RTT for these observations.
        if type(event).__name__ == "ObservationEvent" and event.tool_name in {
            "search",
            "get_document",
            "read_document",
        }:
            observation = event.observation
            if not observation.is_error:
                for item in observation.content:
                    text = getattr(item, "text", "")
                    try:
                        value, _ = json.JSONDecoder().raw_decode(text.lstrip())
                    except ValueError:
                        continue
                    rows = value if isinstance(value, list) else [value]
                    self.retrieved_docids.update(
                        str(row["docid"])
                        for row in rows
                        if isinstance(row, dict) and "docid" in row
                    )
        super().sdk_event(event)

    def close(self):
        self._feedback_queue.put(None)
        self._feedback_thread.join()
        write_json(self.directory / "predictor_transport.json", dict(self.predictor_metrics))
        super().close()
