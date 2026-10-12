"""Native OpenHands conformer pilot; no gateway or tool-result cache."""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from openhands.sdk import Agent, Conversation, ToolDefinition
from openhands.sdk.tool import (
    Action,
    Observation,
    Tool,
    ToolAnnotations,
    ToolExecutor,
    register_tool,
)
from pydantic import Field, SecretStr

from .sdk_bridge import Binding, RecordedLLM, bind, get_binding, unbind
from .tracing import Budget, TraceRecorder, write_json

MOLECULES = {"hexadecane": "C" * 16, "octacosane": "C" * 28}
PARAMETERS = {"num_confs": 128, "seed": 41, "max_iterations": 500}


class ConformerAction(Action):
    molecule_id: str = Field(
        min_length=1,
        description="Exact molecule ID from the task; executor resolves its fixed SMILES",
    )
    num_confs: int = Field(default=128, ge=1, le=512)
    seed: int = Field(default=41, ge=1)
    max_iterations: int = Field(default=500, ge=1, le=1000)


class ConformerObservation(Observation):
    pass


@dataclass(frozen=True)
class ConformerEnvironment:
    molecule_id: str
    python: str


class ConformerExecutor(ToolExecutor):
    def __init__(self, binding: Binding):
        self.binding = binding

    def __call__(self, action: ConformerAction, conversation=None):
        b = self.binding
        environment = b.environment
        assert isinstance(environment, ConformerEnvironment)
        arguments = action.model_dump(mode="json", exclude={"kind", "summary", "security_risk"})
        b.budget.tool()
        link = b.recorder.start_tool("compute_conformers", arguments)
        started = time.monotonic()
        try:
            expected = {"molecule_id": environment.molecule_id, **PARAMETERS}
            if arguments != expected:
                raise ValueError(f"Use exactly the requested parameters: {expected}")
            arguments["smiles"] = MOLECULES[environment.molecule_id]
            with b.recorder.activity("tool"):
                proc = subprocess.run(
                    [environment.python, str(Path(__file__).with_name("rdkit_conformers.py"))],
                    input=json.dumps(arguments),
                    capture_output=True,
                    text=True,
                    timeout=min(b.tool_timeout, b.budget.remaining()),
                    check=True,
                )
            data = json.loads(proc.stdout)
            text = json.dumps(data)
            b.recorder.emit(
                "tool_end",
                tool_name="compute_conformers",
                **link,
                executor_duration_ms=(time.monotonic() - started) * 1000,
                round_trip_ms=(time.monotonic() - started) * 1000,
                model_observation=text,
                effective_arguments=arguments,
                compute_stage_timing={key: data[key] for key in ("embed_s", "optimize_s")},
                timed_out=False,
            )
            return ConformerObservation.from_text(text)
        except Exception as exc:
            b.recorder.emit(
                "tool_error",
                tool_name="compute_conformers",
                **link,
                error_type=type(exc).__name__,
                round_trip_ms=(time.monotonic() - started) * 1000,
                timed_out=isinstance(exc, subprocess.TimeoutExpired),
            )
            if isinstance(exc, ValueError):
                return ConformerObservation.from_text(str(exc), is_error=True)
            raise


class ComputeConformersTool(ToolDefinition):
    name = "compute_conformers"

    @classmethod
    def create(cls, conv_state=None, **params):
        return [
            cls(
                action_type=ConformerAction,
                observation_type=ConformerObservation,
                description="Compute ETKDGv3 conformers and MMFF94s energies on CPU. "
                "Fixed single-thread execution; returns minimum energy and convergence counts.",
                executor=ConformerExecutor(get_binding(params["binding_key"])),
                annotations=ToolAnnotations(readOnlyHint=True),
            )
        ]


register_tool(ComputeConformersTool.name, ComputeConformersTool)


def run_probe(directory, *, molecule_id, python, model, base_url, api_key):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    recorder = TraceRecorder(
        directory, {"task_id": molecule_id, "benchmark": "rdkit-conformer-pilot"}
    )
    recorder.environment_tool_names = {"compute_conformers"}
    budget = Budget(8, 10, 1800)
    key = bind(Binding(ConformerEnvironment(molecule_id, python), recorder, budget, 600, 16000))
    instruction = (
        f"For molecule_id {molecule_id!r} (fixed SMILES {MOLECULES[molecule_id]}), "
        "run compute_conformers with exactly 128 conformers, seed 41 and max_iterations 500. "
        "Report minimum MMFF94s energy and converged count. Keep the final answer below 50 words."
    )
    write_json(
        directory / "public_task.json",
        {
            "instruction": instruction,
            "smiles": MOLECULES[molecule_id],
            "protocol": "local RDKit conformer pilot; not full ChemCrow benchmark",
            "cache_enabled": False,
        },
    )
    conversation = None
    started = time.monotonic()
    recorder.emit("task_start")
    try:
        llm = RecordedLLM(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
            usage_id="actor",
            temperature=1.0,
            top_p=0.95,
            seed=41,
            max_input_tokens=258048,
            max_output_tokens=4096,
            timeout=900,
            num_retries=0,
            native_tool_calling=True,
            stream=False,
            caching_prompt=False,
            litellm_extra_body={
                "top_k": 20,
                "min_p": 0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        llm.attach(recorder, budget)
        agent = Agent(
            llm=llm,
            tools=[Tool(name=ComputeConformersTool.name, params={"binding_key": key})],
            include_default_tools=["FinishTool"],
            condenser=None,
            system_prompt="Use the supplied computational tool. Report numerical results "
            "concisely and call finish. Do not invent energies.",
        )
        conversation = Conversation(
            agent=agent,
            workspace=str(directory / "workspace"),
            callbacks=[recorder.sdk_event],
            max_iteration_per_run=10,
            persistence_dir=str(directory / "sdk_state"),
            visualizer=None,
        )
        conversation.send_message(instruction)
        conversation.run()
        status = conversation.state.execution_status.value
        result = {
            "execution_status": "completed" if status == "finished" else status,
            "final_text": recorder.final_text,
            "jct_s": time.monotonic() - started,
            "tool_calls": recorder.executed_counts["compute_conformers"],
        }
        write_json(directory / "result.json", result)
        return result
    finally:
        try:
            if conversation is not None:
                conversation.close()
        finally:
            recorder.close()
            unbind(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule-id", choices=tuple(MOLECULES), required=True)
    parser.add_argument("--rdkit-python", default=sys.executable)
    parser.add_argument("--model", default="openai/Qwen3.5-27B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_probe(
                args.output,
                molecule_id=args.molecule_id,
                python=args.rdkit_python,
                model=args.model,
                base_url=args.base_url,
                api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
