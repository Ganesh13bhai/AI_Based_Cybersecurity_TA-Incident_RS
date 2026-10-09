from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


ROOT = Path(__file__).resolve().parents[3]

TRAIN_ROOT = ROOT / "data" / "splits" / "cicids2017" / "train"
VAL_ROOT = ROOT / "data" / "splits" / "cicids2017" / "validation"

MODEL_DIR = ROOT / "models" / "cicids2017"
RESULT_DIR = ROOT / "results" / "ml" / "cicids2017"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "logistic_regression_binary.joblib"
RESULT_PATH = RESULT_DIR / "logistic_regression_binary.json"

MAX_TRAIN_ROWS = 500_000
RANDOM_STATE = 42

EXCLUDED_COLUMNS = {
    "Label",
    "canonical_label",
    "binary_label",
    "original_label",
    "attack_cat",
}


def get_feature_columns():

    files = sorted(VAL_ROOT.glob("*.csv"))

    df = pd.read_csv(
        files[0],
        nrows=5
    )

    return [
        column
        for column in df.columns
        if (
            column not in EXCLUDED_COLUMNS
            and pd.api.types.is_numeric_dtype(df[column])
        )
    ]


def load_training_data(feature_columns):

    X_parts = []
    y_parts = []

    total_loaded = 0

    files = sorted(TRAIN_ROOT.glob("*.csv"))

    print("\nLoading training data...")
    print("=" * 70)

    for index, file in enumerate(files, 1):

        if total_loaded >= MAX_TRAIN_ROWS:
            break

        remaining = MAX_TRAIN_ROWS - total_loaded

        df = pd.read_csv(
            file,
            usecols=feature_columns + ["binary_label"],
            nrows=remaining,
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

        valid = X.notna().all(axis=1) & y.notna()

        X = X.loc[valid].astype(np.float32)
        y = y.loc[valid].astype(np.int8)

        X_parts.append(X)
        y_parts.append(y)

        total_loaded += len(X)

        print(
            f"{index:02d}/{len(files)} "
            f"{file.name}: {len(X):,} rows"
        )

    X_train = pd.concat(
        X_parts,
        ignore_index=True
    )

    y_train = pd.concat(
        y_parts,
        ignore_index=True
    )

    print("=" * 70)
    print(f"Training rows: {len(X_train):,}")
    print(f"Features: {X_train.shape[1]}")

    return X_train, y_train


def load_validation_data(feature_columns):

    X_parts = []
    y_parts = []

    files = sorted(VAL_ROOT.glob("*.csv"))

    print("\nLoading validation data...")
    print("=" * 70)

    for index, file in enumerate(files, 1):

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

        valid = X.notna().all(axis=1) & y.notna()

        X_parts.append(
            X.loc[valid].astype(np.float32)
        )

        y_parts.append(
            y.loc[valid].astype(np.int8)
        )

        print(
            f"{index:02d}/{len(files)} "
            f"{file.name}: {valid.sum():,} rows"
        )

    X_val = pd.concat(
        X_parts,
        ignore_index=True
    )

    y_val = pd.concat(
        y_parts,
        ignore_index=True
    )

    print("=" * 70)
    print(f"Validation rows: {len(X_val):,}")

    return X_val, y_val


def main():

    print()
    print("=" * 70)
    print("CIC-IDS2017 LOGISTIC REGRESSION BINARY CLASSIFICATION")
    print("=" * 70)

    feature_columns = get_feature_columns()

    print(
        f"\nNumber of features: {len(feature_columns)}"
    )

    X_train, y_train = load_training_data(
        feature_columns
    )

    X_val, y_val = load_validation_data(
        feature_columns
    )

    print("\nCreating Logistic Regression pipeline...")

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                solver="lbfgs",
                n_jobs=-1,
            )
        )
    ])

    print("\nTraining Logistic Regression...")

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    print(
        f"Training completed in "
        f"{training_time:.2f} seconds"
    )

    print("\nGenerating validation predictions...")

    prediction_start = time.time()

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(np.int8)

    prediction_time = time.time() - prediction_start

    accuracy = accuracy_score(
        y_val,
        predictions
    )

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

    roc_auc = roc_auc_score(
        y_val,
        probabilities
    )

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print()
    print("=" * 70)
    print("LOGISTIC REGRESSION VALIDATION RESULTS")
    print("=" * 70)

    print(f"Accuracy            : {accuracy:.6f}")
    print(f"Precision           : {precision:.6f}")
    print(f"Recall              : {recall:.6f}")
    print(f"F1 Score            : {f1:.6f}")
    print(f"ROC-AUC             : {roc_auc:.6f}")

    print("\nConfusion Matrix")
    print("-" * 50)

    print("                 Predicted")
    print("                 Benign     Attack")
    print(f"Actual Benign    {tn:8,d} {fp:10,d}")
    print(f"Actual Attack    {fn:8,d} {tp:10,d}")

    print()
    print(f"True Negatives     : {tn:,}")
    print(f"False Positives    : {fp:,}")
    print(f"False Negatives    : {fn:,}")
    print(f"True Positives     : {tp:,}")

    print()
    print(f"False Positive Rate: {fpr:.6f}")
    print(f"False Negative Rate: {fnr:.6f}")

    print("\nClassification Report")
    print("-" * 70)

    print(
        classification_report(
            y_val,
            predictions,
            target_names=[
                "BENIGN",
                "ATTACK"
            ],
            zero_division=0
        )
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    report_dict = classification_report(
        y_val,
        predictions,
        target_names=[
            "BENIGN",
            "ATTACK"
        ],
        output_dict=True,
        zero_division=0
    )

    results = {
        "dataset": "CIC-IDS2017",
        "model": "LogisticRegression",
        "task": "binary_detection",

        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "feature_count": int(len(feature_columns)),

        "threshold": 0.50,

        "parameters": {
            "max_iter": 300,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "random_state": RANDOM_STATE,
        },

        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
        },

        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },

        "timing": {
            "training_seconds": float(training_time),
            "prediction_seconds": float(prediction_time),
        },

        "classification_report": report_dict,
    }

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print("\nModel saved:")
    print(MODEL_PATH)

    print("\nResults saved:")
    print(RESULT_PATH)

    print()
    print("=" * 70)
    print("LOGISTIC REGRESSION TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()