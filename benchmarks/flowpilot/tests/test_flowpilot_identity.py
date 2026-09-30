from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

from openhands.sdk import Agent, LocalConversation
from openhands.sdk.flowpilot import FlowPilotRuntime
from openhands.sdk.llm import LLM

from benchmark_adapters.config import Config, DatasetConfig
from benchmark_adapters.predictor_integration import benchmark_flowpilot_config


def test_benchmark_workflows_use_conversation_identity_and_preserve_it_on_resume(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FLOWPILOT_PREDICTOR_GATEWAY", "http://unused")
    monkeypatch.setenv("FLOWPILOT_INGRESS_API_KEY", "fixture-key")
    monkeypatch.delenv("FLOWPILOT_EXPERIMENT_PROFILE", raising=False)
    monkeypatch.delenv("FLOWPILOT_REUSE_ENABLED", raising=False)
    config = Config(dataset=DatasetConfig(kind="hotpot", id="data", revision="v1"))
    agent = Agent(llm=LLM(model="openai/gpt-4o"), tools=[], include_default_tools=[])
    job_ids = []
    with patch.object(FlowPilotRuntime, "register"), ExitStack() as stack:
        for task_id, kind in [("a", "hotpot"), ("b", "hotpot"), ("a", "browsecomp")]:
            adapter = benchmark_flowpilot_config(
                replace(config, dataset=replace(config.dataset, kind=kind)),
                {"run_id": "same-campaign", "attempt_id": "attempt1", "task_id": task_id},
            )
            assert adapter.job_id == adapter.line_id == ""
            conversation = LocalConversation(
                agent=agent,
                workspace=tmp_path,
                persistence_dir=tmp_path / "state",
                flowpilot=adapter,
                visualizer=None,
            )
            stack.callback(conversation.close)
            conversation_id = str(conversation.state.id)
            resolved = conversation._flowpilot
            assert resolved.job_id == f"job-{conversation_id}"
            assert resolved.line_id == f"line-{conversation_id}"
            assert resolved.root_conversation_id == conversation_id
            job_ids.append(resolved.job_id)
            conversation.close()
            resumed = LocalConversation(
                agent=agent,
                workspace=tmp_path,
                persistence_dir=tmp_path / "state",
                conversation_id=conversation.state.id,
                flowpilot=adapter,
                visualizer=None,
            )
            stack.callback(resumed.close)
            assert resumed._flowpilot.job_id == resolved.job_id
            assert resumed._flowpilot.line_id == resolved.line_id
        assert len(set(job_ids)) == 3
