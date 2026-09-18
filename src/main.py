from src.tools import (
    correlation_matrix,
    load_dataset,
    numeric_summary,
    summarize_dataset,
)


DATA_PATH = "data/sample_sales.csv"


df = load_dataset(
    DATA_PATH
)


print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(
    summarize_dataset(
        df
    )
)


print()
print("=" * 60)
print("NUMERIC SUMMARY")
print("=" * 60)

print(
    numeric_summary(
        df
    )
)


print()
print("=" * 60)
print("CORRELATIONS")
print("=" * 60)

print(
    correlation_matrix(
        df
    )
)