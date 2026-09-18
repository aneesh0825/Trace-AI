import json

import pandas as pd
import pytest

from src.agent import execute_tool


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "revenue": [100, 200, 300, None],
            "region": ["west", "east", "west", "east"],
        }
    )


def test_unknown_tool_name_returns_an_error_result(sample_df):
    result = execute_tool("not_a_real_tool", {}, sample_df)

    assert result["is_error"] is True
    assert "not_a_real_tool" in result["content"]


def test_content_is_always_a_string(sample_df):
    result = execute_tool("summarize_dataset", {}, sample_df)

    assert isinstance(result["content"], str)


def test_summarize_dataset_dispatches_with_no_arguments(sample_df):
    result = execute_tool("summarize_dataset", {}, sample_df)

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["rows"] == 4
    assert payload["columns"] == 2
    assert "revenue" in payload["column_names"]


def test_numeric_summary_serializes_numpy_floats_without_crashing(sample_df):
    result = execute_tool("numeric_summary", {}, sample_df)

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["revenue"]["mean"] == pytest.approx(200.0)


def test_value_counts_passes_through_column_argument(sample_df):
    result = execute_tool("value_counts", {"column": "region"}, sample_df)

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["west"] == 2
    assert payload["east"] == 2


def test_value_counts_missing_required_argument_is_an_error(sample_df):
    result = execute_tool("value_counts", {}, sample_df)

    assert result["is_error"] is True
    assert "column" in result["content"]


def test_value_counts_invalid_column_is_an_error_not_a_crash(sample_df):
    result = execute_tool("value_counts", {"column": "does_not_exist"}, sample_df)

    assert result["is_error"] is True
    assert "does_not_exist" in result["content"]


def test_correlation_matrix_dispatches_with_no_arguments(sample_df):
    result = execute_tool("correlation_matrix", {}, sample_df)

    assert result["is_error"] is False
    # Only one numeric column in sample_df, so tools.py returns {}
    assert json.loads(result["content"]) == {}


def test_unexpected_argument_from_a_hallucinating_llm_is_an_error(sample_df):
    result = execute_tool(
        "summarize_dataset", {"made_up_arg": 123}, sample_df
    )

    assert result["is_error"] is True


def test_compare_segment_dispatches_with_arguments(sample_df):
    result = execute_tool(
        "compare_segment",
        {"filter_column": "region", "filter_value": "west"},
        sample_df,
    )

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["revenue"]["segment_mean"] == pytest.approx(200.0)


def test_compare_segment_bad_column_is_an_error_not_a_crash(sample_df):
    result = execute_tool(
        "compare_segment",
        {"filter_column": "does_not_exist", "filter_value": "west"},
        sample_df,
    )

    assert result["is_error"] is True
    assert "does_not_exist" in result["content"]
