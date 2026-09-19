import pandas as pd
import pytest

from src.agent import continue_investigation
from src.state import AgentState
from tests.fakes import FakeClient, response, text_block, tool_use_block


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "revenue": [100, 200, 300, 400],
            "region": ["west", "east", "west", "east"],
        }
    )


def test_appends_follow_up_and_preserves_existing_history(sample_df):
    client = FakeClient(
        [response([text_block("Revised answer.")], stop_reason="end_turn")]
    )
    state = AgentState(
        df=sample_df,
        messages=[
            {"role": "user", "content": "How many rows?"},
            {"role": "assistant", "content": [text_block("There are 4 rows.")]},
        ],
    )

    final_state = continue_investigation(
        client, state, "I don't buy this, check region instead."
    )

    # Original history is untouched...
    assert final_state.messages[0] == {"role": "user", "content": "How many rows?"}
    assert final_state.messages[1]["content"][0].text == "There are 4 rows."
    # ...and the follow-up was appended as a new user message before the next call.
    assert final_state.messages[2] == {
        "role": "user",
        "content": "I don't buy this, check region instead.",
    }
    assert final_state.messages[-1]["content"][0].text == "Revised answer."


def test_follow_up_is_sent_to_claude_as_part_of_the_next_request(sample_df):
    client = FakeClient(
        [response([text_block("Revised answer.")], stop_reason="end_turn")]
    )
    state = AgentState(
        df=sample_df,
        messages=[{"role": "user", "content": "How many rows?"}],
    )

    continue_investigation(client, state, "Check region instead.")

    sent_messages = client.messages.calls[0]["messages"]
    assert sent_messages[-1] == {"role": "user", "content": "Check region instead."}


def test_continue_investigation_can_trigger_new_tool_calls(sample_df):
    client = FakeClient(
        [
            response(
                [tool_use_block("t1", "value_counts", {"column": "region"})],
                stop_reason="tool_use",
            ),
            response(
                [text_block("Region is evenly split.")], stop_reason="end_turn"
            ),
        ]
    )
    state = AgentState(
        df=sample_df,
        messages=[
            {"role": "user", "content": "Investigate revenue."},
            {"role": "assistant", "content": [text_block("Revenue looks flat.")]},
        ],
    )

    final_state = continue_investigation(
        client, state, "Check the region breakdown too."
    )

    assert len(client.messages.calls) == 2
    tool_result_message = final_state.messages[4]
    assert tool_result_message["role"] == "user"
    assert tool_result_message["content"][0]["tool_use_id"] == "t1"
    assert final_state.messages[-1]["content"][0].text == "Region is evenly split."


def test_continue_investigation_passes_through_max_turns(sample_df):
    scripted = [
        response(
            [tool_use_block(f"t{i}", "summarize_dataset", {})], stop_reason="tool_use"
        )
        for i in range(2)
    ]
    client = FakeClient(scripted)
    state = AgentState(df=sample_df, messages=[{"role": "user", "content": "Investigate."}])

    continue_investigation(client, state, "Try again.", max_turns=2)

    assert len(client.messages.calls) == 2
