import json

import pandas as pd
import pytest

from src.agent import execute_tool
from src.state import AgentState


@pytest.fixture
def datasets():
    return {
        "sales": pd.DataFrame(
            {
                "revenue": [100, 200, 300, None],
                "region": ["west", "east", "west", "east"],
            }
        )
    }


@pytest.fixture
def state(datasets):
    return AgentState(datasets=datasets)


def test_unknown_tool_name_returns_an_error_result(state):
    result = execute_tool("not_a_real_tool", {"dataset_name": "sales"}, state)

    assert result["is_error"] is True
    assert "not_a_real_tool" in result["content"]


def test_content_is_always_a_string(state):
    result = execute_tool("summarize_dataset", {"dataset_name": "sales"}, state)

    assert isinstance(result["content"], str)


def test_summarize_dataset_dispatches_with_no_arguments(state):
    result = execute_tool("summarize_dataset", {"dataset_name": "sales"}, state)

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["rows"] == 4
    assert payload["columns"] == 2
    assert "revenue" in payload["column_names"]


def test_numeric_summary_serializes_numpy_floats_without_crashing(state):
    result = execute_tool("numeric_summary", {"dataset_name": "sales"}, state)

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["revenue"]["mean"] == pytest.approx(200.0)


def test_value_counts_passes_through_column_argument(state):
    result = execute_tool(
        "value_counts", {"dataset_name": "sales", "column": "region"}, state
    )

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["west"] == 2
    assert payload["east"] == 2


def test_value_counts_missing_required_argument_is_an_error(state):
    result = execute_tool("value_counts", {"dataset_name": "sales"}, state)

    assert result["is_error"] is True
    assert "column" in result["content"]


def test_value_counts_invalid_column_is_an_error_not_a_crash(state):
    result = execute_tool(
        "value_counts",
        {"dataset_name": "sales", "column": "does_not_exist"},
        state,
    )

    assert result["is_error"] is True
    assert "does_not_exist" in result["content"]


def test_correlation_matrix_dispatches_with_no_arguments(state):
    result = execute_tool("correlation_matrix", {"dataset_name": "sales"}, state)

    assert result["is_error"] is False
    # Only one numeric column in datasets["sales"], so tools.py returns {}
    assert json.loads(result["content"]) == {}


def test_unexpected_argument_from_a_hallucinating_llm_is_an_error(state):
    result = execute_tool(
        "summarize_dataset", {"dataset_name": "sales", "made_up_arg": 123}, state
    )

    assert result["is_error"] is True


def test_compare_segment_dispatches_with_arguments(state):
    result = execute_tool(
        "compare_segment",
        {"dataset_name": "sales", "filter_column": "region", "filter_value": "west"},
        state,
    )

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["revenue"]["segment_mean"] == pytest.approx(200.0)


def test_compare_segment_bad_column_is_an_error_not_a_crash(state):
    result = execute_tool(
        "compare_segment",
        {
            "dataset_name": "sales",
            "filter_column": "does_not_exist",
            "filter_value": "west",
        },
        state,
    )

    assert result["is_error"] is True
    assert "does_not_exist" in result["content"]


def test_unknown_dataset_name_is_an_error_not_a_crash(state):
    result = execute_tool(
        "summarize_dataset", {"dataset_name": "does_not_exist"}, state
    )

    assert result["is_error"] is True
    assert "does_not_exist" in result["content"]
    # Helpful: tells Claude what it could have picked instead.
    assert "sales" in result["content"]


def test_missing_dataset_name_is_an_error_not_a_crash(state):
    result = execute_tool("summarize_dataset", {}, state)

    assert result["is_error"] is True


def test_multiple_datasets_are_kept_independent():
    state = AgentState(
        datasets={
            "sales": pd.DataFrame({"revenue": [100, 200]}),
            "returns": pd.DataFrame({"revenue": [10, 20, 30]}),
        }
    )

    sales_result = execute_tool("summarize_dataset", {"dataset_name": "sales"}, state)
    returns_result = execute_tool(
        "summarize_dataset", {"dataset_name": "returns"}, state
    )

    assert json.loads(sales_result["content"])["rows"] == 2
    assert json.loads(returns_result["content"])["rows"] == 3


def test_run_sql_dispatches_with_dataset_table_alias(state):
    result = execute_tool(
        "run_sql",
        {
            "dataset_name": "sales",
            "query": "SELECT region, COUNT(*) AS n FROM dataset GROUP BY region",
        },
        state,
    )

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    counts = {row["region"]: row["n"] for row in payload["rows"]}
    assert counts == {"west": 2, "east": 2}


def test_run_sql_non_select_is_an_error_not_a_crash(state):
    result = execute_tool(
        "run_sql",
        {"dataset_name": "sales", "query": "DELETE FROM dataset"},
        state,
    )

    assert result["is_error"] is True


def test_run_sql_bad_sql_is_an_error_not_a_crash(state):
    result = execute_tool(
        "run_sql",
        {"dataset_name": "sales", "query": "SELECT does_not_exist FROM dataset"},
        state,
    )

    assert result["is_error"] is True


# --- graph tools (record_hypothesis / update_hypothesis) ---
#
# These don't touch state.datasets at all - they read/write state.graph,
# and turn is stamped by the caller rather than supplied by Claude.


def test_record_hypothesis_dispatches_and_writes_to_state_graph(state):
    result = execute_tool(
        "record_hypothesis",
        {"statement": "August drop is a west region issue."},
        state,
        turn=1,
    )

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["id"] == "h1"
    assert payload["status"] == "proposed"
    assert state.graph == [payload]


def test_record_hypothesis_rejects_empty_statement_as_an_error(state):
    result = execute_tool("record_hypothesis", {"statement": "   "}, state)

    assert result["is_error"] is True


def test_update_hypothesis_dispatches_and_mutates_existing_entry(state):
    execute_tool(
        "record_hypothesis",
        {"statement": "August drop is a west region issue."},
        state,
        turn=1,
    )

    result = execute_tool(
        "update_hypothesis",
        {
            "hypothesis_id": "h1",
            "status": "supported",
            "evidence": {
                "tool_name": "compare_segment",
                "note": "West region revenue down 40% vs baseline.",
            },
        },
        state,
        turn=2,
    )

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["status"] == "supported"
    assert payload["evidence"] == [
        {
            "tool_name": "compare_segment",
            "note": "West region revenue down 40% vs baseline.",
        }
    ]
    assert state.graph[0]["status"] == "supported"


def test_update_hypothesis_unknown_id_is_an_error_not_a_crash(state):
    result = execute_tool(
        "update_hypothesis", {"hypothesis_id": "h99", "status": "supported"}, state
    )

    assert result["is_error"] is True
    assert "h99" in result["content"]


def test_update_hypothesis_invalid_status_is_an_error_not_a_crash(state):
    execute_tool(
        "record_hypothesis", {"statement": "August drop is seasonal."}, state
    )

    result = execute_tool(
        "update_hypothesis", {"hypothesis_id": "h1", "status": "very_sure"}, state
    )

    assert result["is_error"] is True


def test_claude_cannot_override_the_harness_assigned_turn(state):
    result = execute_tool(
        "record_hypothesis",
        {"statement": "August drop is a west region issue.", "turn": 999},
        state,
        turn=1,
    )

    payload = json.loads(result["content"])
    assert payload["created_turn"] == 1
