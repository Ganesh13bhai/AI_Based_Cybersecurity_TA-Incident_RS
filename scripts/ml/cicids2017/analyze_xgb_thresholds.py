from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    ROOT
    / "models"
    / "cicids2017"
    / "xgboost_binary.json"
)

VALIDATION_ROOT = (
    ROOT
    / "data"
    / "splits"
    / "cicids2017"
    / "validation"
)

EXCLUDED_COLUMNS = {
    "Label",
    "canonical_label",
    "binary_label",
    "original_label",
    "attack_cat",
}


def get_feature_columns():

    first_file = sorted(
        VALIDATION_ROOT.glob("*.csv")
    )[0]

    df = pd.read_csv(
        first_file,
        nrows=5
    )

    return [
        column
        for column in df.columns
        if (
            column not in EXCLUDED_COLUMNS
            and pd.api.types.is_numeric_dtype(
                df[column]
            )
        )
    ]


def load_validation(feature_columns):

    X_parts = []
    y_parts = []

    files = sorted(
        VALIDATION_ROOT.glob("*.csv")
    )

    print("Loading validation data...")

    for i, file in enumerate(files, 1):

        print(
            f"{i:02d}/{len(files)} "
            f"{file.name}"
        )

        df = pd.read_csv(
            file,
            usecols=feature_columns + ["binary_label"]
        )

        X = df[feature_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        y = pd.to_numeric(
            df["binary_label"],
            errors="coerce"
        )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        valid = (
            X.notna().all(axis=1)
            & y.notna()
        )

        X_parts.append(
            X.loc[valid].astype(np.float32)
        )

        y_parts.append(
            y.loc[valid].astype(np.int8)
        )

    return (
        pd.concat(
            X_parts,
            ignore_index=True
        ),
        pd.concat(
            y_parts,
            ignore_index=True
        )
    )


def main():

    print("=" * 60)
    print("XGBOOST THRESHOLD ANALYSIS")
    print("=" * 60)

    feature_columns = get_feature_columns()

    model = xgb.XGBClassifier()

    model.load_model(
        MODEL_PATH
    )

    X_val, y_val = load_validation(
        feature_columns
    )

    print()
    print("Generating attack probabilities...")

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    print()
    print("=" * 75)

    print(
        f"{'Threshold':>10}"
        f"{'Precision':>15}"
        f"{'Recall':>15}"
        f"{'F1':>15}"
    )

    print("=" * 75)

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(np.int8)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        print(
            f"{threshold:>10.2f}"
            f"{precision:>15.4f}"
            f"{recall:>15.4f}"
            f"{f1:>15.4f}"
        )


if __name__ == "__main__":
    main()