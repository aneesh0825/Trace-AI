import pandas as pd
import pytest

from src.tools import compare_segment, run_sql


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "month": ["Jul", "Jul", "Aug", "Aug", "Sep", "Sep"],
            "revenue": [100, 120, 40, 50, 110, 130],
            "orders": [10, 12, 8, 9, 11, 13],
        }
    )


def test_missing_filter_column_raises(sample_df):
    with pytest.raises(ValueError):
        compare_segment(sample_df, "nonexistent", "Aug")


def test_filter_value_with_no_matching_rows_raises(sample_df):
    with pytest.raises(ValueError):
        compare_segment(sample_df, "month", "Dec")


def test_filter_value_matching_every_row_raises():
    df = pd.DataFrame({"month": ["Aug", "Aug"], "revenue": [1, 2]})

    with pytest.raises(ValueError):
        compare_segment(df, "month", "Aug")


def test_compare_segment_computes_expected_stats(sample_df):
    result = compare_segment(sample_df, "month", "Aug")

    # Non-numeric columns (like "month" itself) aren't part of the result.
    assert set(result.keys()) == {"revenue", "orders"}

    assert result["revenue"]["segment_mean"] == pytest.approx(45.0)
    assert result["revenue"]["baseline_mean"] == pytest.approx(115.0)
    assert result["revenue"]["difference"] == pytest.approx(-70.0)
    assert result["revenue"]["percent_change"] == pytest.approx(-60.87, rel=1e-2)


def test_percent_change_is_none_when_baseline_mean_is_zero():
    df = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "value": [10, 10, 0, 0],
        }
    )

    result = compare_segment(df, "group", "a")

    assert result["value"]["baseline_mean"] == 0
    assert result["value"]["percent_change"] is None


def test_run_sql_basic_filter(sample_df):
    result = run_sql(
        sample_df, "SELECT month, revenue FROM dataset WHERE month = 'Aug'"
    )

    assert result["row_count"] == 2
    assert result["truncated"] is False
    assert {row["revenue"] for row in result["rows"]} == {40, 50}


def test_run_sql_aggregation(sample_df):
    result = run_sql(
        sample_df,
        "SELECT month, SUM(revenue) AS total FROM dataset "
        "GROUP BY month ORDER BY month",
    )

    totals = {row["month"]: row["total"] for row in result["rows"]}
    assert totals == {"Aug": 90, "Jul": 220, "Sep": 240}


def test_run_sql_allows_with_cte(sample_df):
    result = run_sql(
        sample_df,
        "WITH totals AS (SELECT month, revenue FROM dataset) "
        "SELECT * FROM totals WHERE revenue > 100",
    )

    assert result["row_count"] == 3


def test_run_sql_allows_single_trailing_semicolon(sample_df):
    result = run_sql(sample_df, "SELECT * FROM dataset;")

    assert result["row_count"] == 6


def test_run_sql_rejects_empty_query(sample_df):
    with pytest.raises(ValueError):
        run_sql(sample_df, "   ")


def test_run_sql_rejects_non_select_statements(sample_df):
    with pytest.raises(ValueError):
        run_sql(sample_df, "DROP TABLE dataset")


def test_run_sql_rejects_chained_statements(sample_df):
    with pytest.raises(ValueError):
        run_sql(sample_df, "SELECT 1; DROP TABLE dataset;")


def test_run_sql_bad_column_raises(sample_df):
    with pytest.raises(Exception):
        run_sql(sample_df, "SELECT nonexistent_column FROM dataset")


def test_run_sql_truncates_large_results():
    df = pd.DataFrame({"n": range(500)})

    result = run_sql(df, "SELECT * FROM dataset")

    assert result["row_count"] == 500
    assert result["truncated"] is True
    assert len(result["rows"]) == 200
