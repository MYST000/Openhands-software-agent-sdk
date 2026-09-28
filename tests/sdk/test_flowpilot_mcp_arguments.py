import json

import pytest

from openhands.sdk.event import ActionEvent
from openhands.sdk.flowpilot import _execution_tool_arguments
from openhands.sdk.llm import MessageToolCall
from openhands.sdk.mcp.definition import MCPToolAction


@pytest.mark.parametrize("name", ["search", "corpus_search", "tavily-search"])
def test_flowpilot_uses_native_mcp_arguments_for_any_tool_name(name: str) -> None:
    arguments = {"query": 'Who wrote "Book A"?'}
    event = ActionEvent(
        thought=[],
        action=MCPToolAction(data=arguments),
        tool_name=name,
        tool_call_id="call-1",
        tool_call=MessageToolCall(
            id="call-1", name=name, arguments=json.dumps(arguments), origin="completion"
        ),
        llm_response_id="response-1",
    )
    assert _execution_tool_arguments(event) == arguments
