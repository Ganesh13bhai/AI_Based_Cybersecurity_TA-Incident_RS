from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

SPLIT_ROOT = ROOT / "data" / "splits" / "cicids2017"
MODEL_DIR = ROOT / "models" / "cicids2017"
RESULT_DIR = ROOT / "results" / "ml" / "cicids2017"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "binary_label"

EXCLUDED_COLUMNS = {
    "Label",
    "canonical_label",
    "binary_label",
    "original_label",
    "attack_cat",
}

# Controlled training sample.
# We deliberately do not load all 1.8M rows into RAM.
MAX_TRAIN_ROWS = 500_000

RANDOM_STATE = 42

N_ESTIMATORS = 200

N_JOBS = -1

CHUNK_SIZE = 100_000


# ============================================================
# DISCOVER FEATURES
# ============================================================

def get_feature_columns():

    train_dir = SPLIT_ROOT / "train"

    first_file = sorted(train_dir.glob("*.csv"))[0]

    df = pd.read_csv(first_file, nrows=5)

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    return feature_columns


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data(feature_columns):

    train_dir = SPLIT_ROOT / "train"

    files = sorted(train_dir.glob("*.csv"))

    X_parts = []
    y_parts = []

    rows_loaded = 0

    print()
    print("Loading training data")
    print("=" * 60)

    for file in files:

        if rows_loaded >= MAX_TRAIN_ROWS:
            break

        remaining = MAX_TRAIN_ROWS - rows_loaded

        read_rows = min(CHUNK_SIZE, remaining)

        print(
            f"Reading {file.name} "
            f"(up to {read_rows:,} rows)"
        )

        df = pd.read_csv(
            file,
            usecols=feature_columns + [TARGET],
            nrows=read_rows,
        )

        X = df[feature_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        y = pd.to_numeric(
            df[TARGET],
            errors="coerce"
        )

        # Replace infinities.
        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Remove invalid rows.
        valid = (
            X.notna().all(axis=1)
            & y.notna()
        )

        X = X.loc[valid]
        y = y.loc[valid]

        # float32 significantly reduces RAM usage.
        X = X.astype(np.float32)

        y = y.astype(np.int8)

        X_parts.append(X)
        y_parts.append(y)

        rows_loaded += len(X)

        print(
            f"  Valid rows: {len(X):,}"
        )

    X_train = pd.concat(
        X_parts,
        ignore_index=True
    )

    y_train = pd.concat(
        y_parts,
        ignore_index=True
    )

    print()
    print("Training data loaded")
    print("-" * 60)
    print("Rows:", f"{len(X_train):,}")
    print("Features:", len(X_train.columns))
    print()
    print("Class distribution:")

    print(
        y_train.value_counts()
        .sort_index()
    )

    return X_train, y_train


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_validation_data(feature_columns):

    validation_dir = SPLIT_ROOT / "validation"

    files = sorted(validation_dir.glob("*.csv"))

    X_parts = []
    y_parts = []

    print()
    print("Loading validation data")
    print("=" * 60)

    for file in files:

        print(f"Reading {file.name}")

        df = pd.read_csv(
            file,
            usecols=feature_columns + [TARGET],
        )

        X = df[feature_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        y = pd.to_numeric(
            df[TARGET],
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

        X = X.loc[valid].astype(np.float32)
        y = y.loc[valid].astype(np.int8)

        X_parts.append(X)
        y_parts.append(y)

    X_val = pd.concat(
        X_parts,
        ignore_index=True
    )

    y_val = pd.concat(
        y_parts,
        ignore_index=True
    )

    print()
    print("Validation data loaded")
    print("-" * 60)
    print("Rows:", f"{len(X_val):,}")
    print("Features:", len(X_val.columns))

    return X_val, y_val


# ============================================================
# TRAIN
# ============================================================

def train():

    print()
    print("=" * 60)
    print("CIC-IDS2017 RANDOM FOREST")
    print("Binary Attack Detection")
    print("=" * 60)

    feature_columns = get_feature_columns()

    print()
    print("Number of features:", len(feature_columns))

    X_train, y_train = load_training_data(
        feature_columns
    )

    X_val, y_val = load_validation_data(
        feature_columns
    )

    print()
    print("=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        class_weight="balanced_subsample",
        max_features="sqrt",
        min_samples_leaf=1,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.perf_counter() - start

    print()
    print(
        f"Training completed in "
        f"{training_time:.2f} seconds"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)

    start = time.perf_counter()

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    prediction_time = time.perf_counter() - start

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        y_prob
    )

    cm = confusion_matrix(
        y_val,
        y_pred
    )

    print()
    print(f"Accuracy : {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1       : {f1:.6f}")
    print(f"ROC-AUC  : {roc_auc:.6f}")

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=[
                "BENIGN",
                "ATTACK"
            ],
            zero_division=0
        )
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        MODEL_DIR /
        "random_forest_binary.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {
        "dataset": "CIC-IDS2017",
        "task": "binary_attack_detection",
        "model": "RandomForestClassifier",

        "features": len(feature_columns),

        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),

        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,

        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,

        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),

        "confusion_matrix": cm.tolist(),

        "feature_columns": feature_columns,
    }

    result_path = (
        RESULT_DIR /
        "random_forest_binary.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print("=" * 60)
    print("SAVED")
    print("=" * 60)

    print("Model:")
    print(model_path)

    print()
    print("Results:")
    print(result_path)


if __name__ == "__main__":
    train()