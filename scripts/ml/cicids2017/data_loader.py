from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_ROOT = PROJECT_ROOT / "data" / "splits" / "cicids2017"

TARGET_COLUMNS = {
    "Label",
    "canonical_label",
    "binary_label",
    "original_label",
    "attack_cat",
}


def get_split_files(split: str):
    """Return all CSV files for a dataset split."""
    split_dir = SPLIT_ROOT / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    files = sorted(split_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {split_dir}"
        )

    return files


def inspect_schema(split: str = "train"):
    """Inspect the first CSV file and determine ML feature columns."""

    files = get_split_files(split)

    first_file = files[0]

    df = pd.read_csv(first_file, nrows=5)

    feature_columns = [
        column
        for column in df.columns
        if column not in TARGET_COLUMNS
    ]

    numeric_feature_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    return {
        "files": files,
        "total_columns": len(df.columns),
        "feature_columns": feature_columns,
        "numeric_feature_columns": numeric_feature_columns,
        "target_columns": [
            column
            for column in df.columns
            if column in TARGET_COLUMNS
        ],
    }


def load_csv_chunk(
    file_path: Path,
    feature_columns: list[str],
    target_column: str = "binary_label",
    chunk_size: int = 100_000,
):
    """Load one CSV file in memory-safe chunks."""

    usecols = feature_columns + [target_column]

    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=chunk_size,
    ):
        X = chunk[feature_columns]
        y = chunk[target_column]

        # Convert numeric data safely.
        X = X.apply(pd.to_numeric, errors="coerce")

        # Replace invalid numeric values.
        X = X.replace([np.inf, -np.inf], np.nan)

        # Remove rows containing invalid feature values.
        valid_mask = X.notna().all(axis=1)

        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        yield X, y


if __name__ == "__main__":

    schema = inspect_schema("train")

    print("CIC-IDS2017 ML Schema")
    print("=" * 50)

    print("CSV files:", len(schema["files"]))
    print("Total columns:", schema["total_columns"])
    print("Feature columns:", len(schema["feature_columns"]))
    print(
        "Numeric feature columns:",
        len(schema["numeric_feature_columns"])
    )

    print("\nTarget columns:")
    for column in schema["target_columns"]:
        print("  -", column)

    print("\nFirst 10 ML features:")
    for column in schema["numeric_feature_columns"][:10]:
        print("  -", column)

    print("\nLast 10 ML features:")
    for column in schema["numeric_feature_columns"][-10:]:
        print("  -", column)