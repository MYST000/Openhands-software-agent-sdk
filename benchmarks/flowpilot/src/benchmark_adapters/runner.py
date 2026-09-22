import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from .code_tasks import CODE_KINDS, export_code
from .environment import DockerEnvironment
from .swe import SWEAdapter
from .tracing import Budget, BudgetExceeded, TraceRecorder, write_json


def run_task(
    config,
    task,
    attempt_dir,
    *,
    environment=None,
    run_id=None,
    trace_context=None,
    load_monitor=None,
):
    from openhands.sdk import Agent, Conversation
    from pydantic import SecretStr

    from .sdk_bridge import Binding, RecordedLLM, bind, swe_tools, unbind

    is_code = config.dataset.kind in CODE_KINDS
    if is_code and environment is None:
        raise ValueError("Code benchmarks require an explicitly prepared local environment")
    attempt_dir = Path(attempt_dir)
    if attempt_dir.exists() and any(attempt_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite existing attempt: {attempt_dir}")
    attempt_dir.mkdir(parents=True, exist_ok=True)
    identity = dict(
        task_id=task.task_id,
        dataset_id=task.dataset_id,
        dataset_revision=task.revision,
        split=task.split,
        attempt_id=attempt_dir.name,
        run_id=run_id or uuid.uuid4().hex,
    )
    extra = trace_context or {}
    allowed = {
        "episode_id",
        "replica_id",
        "worker_slot",
        "research_split",
        "task_group_id",
        "queue_position",
        "campaign_partition",
    }
    if extra.keys() - allowed:
        raise ValueError("Unsupported trace context keys")
    identity.update(extra)
    identity["host_id"] = socket.gethostname()
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = "unknown-boot"
    identity["clock_domain"] = f"controller-monotonic:{identity['host_id']}:{boot}"
    recorder = TraceRecorder(attempt_dir, identity, load_monitor=load_monitor)
    recorder.tool_execution_profile = {
        "tool_timeout_seconds": config.runtime.tool_timeout,
        "max_output_chars": (
            config.runtime.max_output_chars if is_code or config.dataset.kind == "swe" else None
        ),
        "intra_task_tool_concurrency": 1,
    }
    write_json(attempt_dir / "public_task.json", task.to_dict())
    write_json(attempt_dir / "profile.json", config.to_dict())
    budget = Budget(
        config.runtime.max_tool_calls, config.runtime.max_llm_requests, config.runtime.task_timeout
    )
    env: Any = environment
    if env is None:
        env = (
            DockerEnvironment(config, task, attempt_dir / "artifacts", identity=identity)
            if config.dataset.kind == "swe"
            else None
        )
    result: dict[str, Any] = dict(
        **identity,
        execution_status="environment_error",
        artifact_status="export_failed",
        evaluation_status="pending",
        termination_reason=None,
        trace_quality="incomplete",
    )
    conv = None
    binding_key = None
    prepared = False
    start = time.monotonic()
    recorder.emit("task_start")
    try:
        if config.dataset.kind == "swe" or is_code:
            metadata = env.prepare()
            prepared = True
            for key in ("environment_id", "container_id"):
                if key in metadata:
                    recorder.identity[key] = result[key] = metadata[key]
            recorder.emit("environment_ready", metadata=metadata)
        else:
            from .native_browsecomp import create_retrieval_environment

            if env is None:
                env = create_retrieval_environment(config)
            metadata = env.prepare()
            prepared = True
            recorder.emit("environment_ready", metadata=metadata)
        result["execution_status"] = "llm_error"
        extra_body = {
            name: getattr(config.llm, name)
            for name in ("top_k", "presence_penalty", "min_p", "repetition_penalty")
            if getattr(config.llm, name) is not None
        }
        if config.llm.enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": config.llm.enable_thinking}
        llm = RecordedLLM(
            model=config.llm.model,
            base_url=config.llm.base_url,
            api_key=SecretStr(os.environ.get(config.llm.api_key_env, "dummy")),
            usage_id="actor",
            temperature=config.llm.temperature,
            top_p=config.llm.top_p,
            seed=config.llm.seed,
            litellm_extra_body=extra_body,
            max_output_tokens=config.llm.max_output_tokens,
            timeout=config.llm.timeout,
            num_retries=config.llm.num_retries,
            native_tool_calling=config.llm.native_tool_calling,
            stream=False,
            caching_prompt=False,
            log_completions=False,
        )
        llm.attach(recorder, budget)
        binding_key = bind(
            Binding(
                env, recorder, budget, config.runtime.tool_timeout, config.runtime.max_output_chars
            )
        )
        if config.dataset.kind == "swe" or is_code:
            from .sdk_bridge import code_tools

            tools = code_tools(binding_key) if is_code else swe_tools(binding_key)
            system = (
                "You are an autonomous software engineering assistant. Use only the supplied workspace tools "
                "to inspect and fix the issue. Follow their stateless-shell and file-editing contracts. "
                "Keep all task changes inside the provided repository. When finished, call finish with a summary."
            )
        else:
            from .retrieval import retrieval_tools

            tools = retrieval_tools(binding_key)
            system = (
                "Answer the question using only the fixed corpus tools. Search and read evidence as needed. "
                "Follow the task output format exactly. Call finish with your final answer."
            )
        recorder.environment_tool_names = {tool.name for tool in tools}
        agent = Agent(
            llm=llm,
            tools=tools,
            system_prompt=system,
            condenser=None,
            include_default_tools=["FinishTool", "ThinkTool"],
            tool_concurrency_limit=1,
        )
        workspace = attempt_dir / "conversation_workspace"
        workspace.mkdir()
        conv = Conversation(
            agent=agent,
            workspace=str(workspace),
            callbacks=[recorder.sdk_event],
            max_iteration_per_run=config.runtime.max_iterations,
            persistence_dir=str(attempt_dir / "sdk_state"),
            visualizer=None,
        )
        recorder.identity["conversation_id"] = str(conv.state.id)
        result["conversation_id"] = str(conv.state.id)
        recorder.emit("conversation_start")
        conv.send_message(task.instruction)
        conv.run()
        budget.check()
        state = conv.state.execution_status.value
        result["sdk_execution_status"] = state
        if budget.reason:
            result["execution_status"] = "budget_exhausted"
            result["termination_reason"] = budget.reason
        elif "MaxIterationsReached" in recorder.error_codes:
            result["execution_status"] = "budget_exhausted"
            result["termination_reason"] = "max_iterations"
        elif state == "finished":
            result["execution_status"] = "completed"
            result["termination_reason"] = "finished"
        else:
            result["execution_status"] = "agent_error"
            result["termination_reason"] = state
        result["trace_quality"] = "recorded"
    except (Exception, KeyboardInterrupt) as exc:
        try:
            budget.check()
        except BudgetExceeded:
            pass
        result["error_type"] = type(exc).__name__
        result["termination_reason"] = budget.reason or type(exc).__name__
        if isinstance(exc, KeyboardInterrupt):
            result["execution_status"] = "cancelled"
        elif budget.reason:
            result["execution_status"] = "budget_exhausted"
        recorder.emit(
            "task_error", error_type=type(exc).__name__, reason=result["termination_reason"]
        )
    finally:
        recorder.emit("conversation_end", execution_status=result["execution_status"])
        if prepared:
            try:
                env.quiesce()
                if config.dataset.kind == "swe":
                    patch = env.export_patch()
                    (attempt_dir / "artifacts").mkdir(exist_ok=True)
                    (attempt_dir / "artifacts" / "model.patch").write_text(patch)
                    submission = SWEAdapter.submission(task, patch, config.llm.model)
                    result["artifact_status"] = "valid" if patch else "empty"
                elif is_code:
                    code = export_code(env, task.public_metadata["solution_path"])
                    (attempt_dir / "artifacts").mkdir(exist_ok=True)
                    (attempt_dir / "artifacts" / "solution.py").write_text(code)
                    submission = dict(
                        task_id=task.task_id,
                        dataset_id=task.dataset_id,
                        code=code,
                        model=config.llm.model,
                    )
                    if config.dataset.kind == "livecodebench":
                        submission = dict(question_id=task.task_id, code_list=[code])
                    result["artifact_status"] = "valid" if code.strip() else "empty"
                else:
                    from .retrieval import export_answer

                    submission = export_answer(config.dataset.kind, task, recorder, result, env)
                    result["artifact_status"] = "valid"
                write_json(attempt_dir / "submission.json", submission)
                recorder.emit(
                    "artifact_exported", path="submission.json", status=result["artifact_status"]
                )
            except Exception as exc:
                result["artifact_status"] = "export_failed"
                result["export_error_type"] = type(exc).__name__
                recorder.emit("artifact_export_error", error_type=type(exc).__name__)
        if conv is not None:
            try:
                conv.close()
            except Exception as exc:
                result["conversation_close_error"] = type(exc).__name__
        if binding_key:
            unbind(binding_key)
        if env is not None:
            try:
                cleanup = None
                for cleanup_attempt in range(2):
                    try:
                        cleanup = env.close()
                        break
                    except Exception:
                        if cleanup_attempt == 1:
                            raise
                recorder.emit("environment_cleanup", metadata=cleanup, cleanup_status="completed")
            except Exception as exc:
                result["environment_cleanup_error"] = type(exc).__name__
                recorder.emit(
                    "environment_cleanup",
                    metadata=getattr(env, "metadata", None),
                    cleanup_status="failed",
                    error_type=type(exc).__name__,
                )
        result.update(
            duration_s=time.monotonic() - start,
            tool_calls=budget.tools,
            llm_requests=budget.requests,
            final_text=recorder.final_text,
            tool_call_counts=dict(recorder.executed_counts),
            errors=recorder.error_codes,
        )
        write_json(attempt_dir / "result.json", result)
        recorder.emit("task_end", result=result)
        recorder.close()
    return result
