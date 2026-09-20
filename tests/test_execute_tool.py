import json

import pandas as pd
import pytest

from src.agent import execute_tool


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


def test_unknown_tool_name_returns_an_error_result(datasets):
    result = execute_tool("not_a_real_tool", {"dataset_name": "sales"}, datasets)

    assert result["is_error"] is True
    assert "not_a_real_tool" in result["content"]


def test_content_is_always_a_string(datasets):
    result = execute_tool("summarize_dataset", {"dataset_name": "sales"}, datasets)

    assert isinstance(result["content"], str)


def test_summarize_dataset_dispatches_with_no_arguments(datasets):
    result = execute_tool("summarize_dataset", {"dataset_name": "sales"}, datasets)

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["rows"] == 4
    assert payload["columns"] == 2
    assert "revenue" in payload["column_names"]


def test_numeric_summary_serializes_numpy_floats_without_crashing(datasets):
    result = execute_tool("numeric_summary", {"dataset_name": "sales"}, datasets)

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["revenue"]["mean"] == pytest.approx(200.0)


def test_value_counts_passes_through_column_argument(datasets):
    result = execute_tool(
        "value_counts", {"dataset_name": "sales", "column": "region"}, datasets
    )

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["west"] == 2
    assert payload["east"] == 2


def test_value_counts_missing_required_argument_is_an_error(datasets):
    result = execute_tool("value_counts", {"dataset_name": "sales"}, datasets)

    assert result["is_error"] is True
    assert "column" in result["content"]


def test_value_counts_invalid_column_is_an_error_not_a_crash(datasets):
    result = execute_tool(
        "value_counts",
        {"dataset_name": "sales", "column": "does_not_exist"},
        datasets,
    )

    assert result["is_error"] is True
    assert "does_not_exist" in result["content"]


def test_correlation_matrix_dispatches_with_no_arguments(datasets):
    result = execute_tool("correlation_matrix", {"dataset_name": "sales"}, datasets)

    assert result["is_error"] is False
    # Only one numeric column in datasets["sales"], so tools.py returns {}
    assert json.loads(result["content"]) == {}


def test_unexpected_argument_from_a_hallucinating_llm_is_an_error(datasets):
    result = execute_tool(
        "summarize_dataset", {"dataset_name": "sales", "made_up_arg": 123}, datasets
    )

    assert result["is_error"] is True


def test_compare_segment_dispatches_with_arguments(datasets):
    result = execute_tool(
        "compare_segment",
        {"dataset_name": "sales", "filter_column": "region", "filter_value": "west"},
        datasets,
    )

    assert result["is_error"] is False
    payload = json.loads(result["content"])
    assert payload["revenue"]["segment_mean"] == pytest.approx(200.0)


def test_compare_segment_bad_column_is_an_error_not_a_crash(datasets):
    result = execute_tool(
        "compare_segment",
        {
            "dataset_name": "sales",
            "filter_column": "does_not_exist",
            "filter_value": "west",
        },
        datasets,
    )

    assert result["is_error"] is True
    assert "does_not_exist" in result["content"]


def test_unknown_dataset_name_is_an_error_not_a_crash(datasets):
    result = execute_tool(
        "summarize_dataset", {"dataset_name": "does_not_exist"}, datasets
    )

    assert result["is_error"] is True
    assert "does_not_exist" in result["content"]
    # Helpful: tells Claude what it could have picked instead.
    assert "sales" in result["content"]


def test_missing_dataset_name_is_an_error_not_a_crash(datasets):
    result = execute_tool("summarize_dataset", {}, datasets)

    assert result["is_error"] is True


def test_multiple_datasets_are_kept_independent():
    datasets = {
        "sales": pd.DataFrame({"revenue": [100, 200]}),
        "returns": pd.DataFrame({"revenue": [10, 20, 30]}),
    }

    sales_result = execute_tool(
        "summarize_dataset", {"dataset_name": "sales"}, datasets
    )
    returns_result = execute_tool(
        "summarize_dataset", {"dataset_name": "returns"}, datasets
    )

    assert json.loads(sales_result["content"])["rows"] == 2
    assert json.loads(returns_result["content"])["rows"] == 3
