import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset from disk.
    """

    df = pd.read_csv(file_path)

    return df


def summarize_dataset(df: pd.DataFrame) -> dict:
    """
    Return basic information about a dataset.
    """

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "missing_values": df.isna().sum().to_dict(),
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
    }

    return summary


def numeric_summary(df: pd.DataFrame) -> dict:
    """
    Return summary statistics for numeric columns.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.empty:
        return {}

    return (
        numeric_df
        .describe()
        .round(3)
        .to_dict()
    )


def value_counts(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
) -> dict:
    """
    Return the most common values in a column.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    counts = (
        df[column]
        .value_counts(
            dropna=False
        )
        .head(top_n)
    )

    return counts.to_dict()


def correlation_matrix(
    df: pd.DataFrame,
) -> dict:
    """
    Return correlations between numeric columns.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] < 2:
        return {}

    correlations = (
        numeric_df
        .corr()
        .round(3)
    )

    return correlations.to_dict()


def compare_segment(
    df: pd.DataFrame,
    filter_column: str,
    filter_value,
) -> dict:
    """
    Compare a segment of the data (rows where filter_column ==
    filter_value) against the rest of the data, for every numeric column.
    """

    if filter_column not in df.columns:
        raise ValueError(
            f"Column '{filter_column}' does not exist."
        )

    segment = df[df[filter_column] == filter_value]
    baseline = df[df[filter_column] != filter_value]

    if segment.empty:
        raise ValueError(
            f"No rows found where '{filter_column}' == {filter_value!r}."
        )

    if baseline.empty:
        raise ValueError(
            f"No rows found outside '{filter_column}' == {filter_value!r}; "
            "nothing to compare against."
        )

    numeric_columns = df.select_dtypes(include="number").columns

    comparison = {}

    for column in numeric_columns:
        segment_mean = segment[column].mean()
        baseline_mean = baseline[column].mean()
        difference = segment_mean - baseline_mean

        if baseline_mean == 0:
            percent_change = None
        else:
            percent_change = round((difference / baseline_mean) * 100, 3)

        comparison[column] = {
            "segment_mean": round(segment_mean, 3),
            "baseline_mean": round(baseline_mean, 3),
            "difference": round(difference, 3),
            "percent_change": percent_change,
        }

    return comparison