import pandas as pd
import pytest

from src.tools import compare_segment


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
