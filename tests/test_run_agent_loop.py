import json

import pandas as pd
import pytest

from src.agent import run_agent_loop
from src.state import AgentState
from tests.fakes import FakeClient, response, text_block, tool_use_block


@pytest.fixture
def datasets():
    return {
        "sales": pd.DataFrame(
            {
                "revenue": [100, 200, 300, 400],
                "region": ["west", "east", "west", "east"],
            }
        )
    }


def test_stops_immediately_when_claude_answers_with_no_tool_use(datasets):
    client = FakeClient(
        [response([text_block("There are 4 rows.")], stop_reason="end_turn")]
    )
    state = AgentState(
        datasets=datasets,
        messages=[{"role": "user", "content": "How many rows?"}],
    )

    final_state = run_agent_loop(client, state)

    assert len(client.messages.calls) == 1
    assert final_state.messages[-1]["role"] == "assistant"
    assert final_state.messages[-1]["content"][0].text == "There are 4 rows."


def test_executes_a_single_tool_call_then_stops(datasets):
    client = FakeClient(
        [
            response(
                [
                    tool_use_block(
                        "t1", "summarize_dataset", {"dataset_name": "sales"}
                    )
                ],
                stop_reason="tool_use",
            ),
            response([text_block("The dataset has 4 rows.")], stop_reason="end_turn"),
        ]
    )
    state = AgentState(
        datasets=datasets,
        messages=[{"role": "user", "content": "How many rows?"}],
    )

    final_state = run_agent_loop(client, state)

    assert len(client.messages.calls) == 2

    # messages: [user question, assistant tool_use, user tool_result, assistant final]
    tool_result_message = final_state.messages[2]
    assert tool_result_message["role"] == "user"
    tool_result_block = tool_result_message["content"][0]
    assert tool_result_block["type"] == "tool_result"
    assert tool_result_block["tool_use_id"] == "t1"
    assert tool_result_block["is_error"] is False
    payload = json.loads(tool_result_block["content"])
    assert payload["rows"] == 4

    assert final_state.messages[-1]["content"][0].text == "The dataset has 4 rows."


def test_multiple_tool_calls_in_one_turn_are_sent_back_in_a_single_message(datasets):
    client = FakeClient(
        [
            response(
                [
                    tool_use_block(
                        "t1", "summarize_dataset", {"dataset_name": "sales"}
                    ),
                    tool_use_block(
                        "t2", "numeric_summary", {"dataset_name": "sales"}
                    ),
                ],
                stop_reason="tool_use",
            ),
            response([text_block("Done.")], stop_reason="end_turn"),
        ]
    )
    state = AgentState(datasets=datasets, messages=[{"role": "user", "content": "Investigate."}])

    final_state = run_agent_loop(client, state)

    tool_result_message = final_state.messages[2]
    assert tool_result_message["role"] == "user"
    assert len(tool_result_message["content"]) == 2
    returned_ids = {block["tool_use_id"] for block in tool_result_message["content"]}
    assert returned_ids == {"t1", "t2"}


def test_stops_after_max_turns_instead_of_looping_forever(datasets):
    # Claude keeps calling tools and never gives a final answer.
    scripted = [
        response(
            [tool_use_block(f"t{i}", "summarize_dataset", {"dataset_name": "sales"})],
            stop_reason="tool_use",
        )
        for i in range(3)
    ]
    client = FakeClient(scripted)
    state = AgentState(datasets=datasets, messages=[{"role": "user", "content": "Investigate."}])

    run_agent_loop(client, state, max_turns=3)

    assert len(client.messages.calls) == 3


def test_a_bad_tool_call_becomes_an_error_result_instead_of_crashing(datasets):
    client = FakeClient(
        [
            response(
                [
                    tool_use_block(
                        "t1", "not_a_real_tool", {"dataset_name": "sales"}
                    )
                ],
                stop_reason="tool_use",
            ),
            response(
                [text_block("I couldn't find that tool.")],
                stop_reason="end_turn",
            ),
        ]
    )
    state = AgentState(datasets=datasets, messages=[{"role": "user", "content": "Investigate."}])

    final_state = run_agent_loop(client, state)

    tool_result_block = final_state.messages[2]["content"][0]
    assert tool_result_block["is_error"] is True
    assert "not_a_real_tool" in tool_result_block["content"]
    # The loop kept going after the bad call instead of raising.
    assert final_state.messages[-1]["content"][0].text == "I couldn't find that tool."


def test_system_prompt_lists_available_dataset_names(datasets):
    client = FakeClient(
        [response([text_block("Done.")], stop_reason="end_turn")]
    )
    state = AgentState(datasets=datasets, messages=[{"role": "user", "content": "Hi."}])

    run_agent_loop(client, state)

    system_prompt = client.messages.calls[0]["system"]
    assert "sales" in system_prompt
