import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


TRAIN_FILE = os.path.join(
    PROCESSED_DIR,
    "selected_train_data.csv"
)

TEST_FILE = os.path.join(
    PROCESSED_DIR,
    "selected_test_data.csv"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

SCALER_FILE = os.path.join(
    PROCESSED_DIR,
    "scaler.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_names.txt"
)

METRICS_FILE = os.path.join(
    RESULT_DIR,
    "binary_model_metrics.txt"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RETRAINING BINARY NIDS MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Load training and testing data
    # --------------------------------------------------------

    print("\nLoading training data...")

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    print(
        f"Training rows: {len(train_df):,}"
    )

    print("\nLoading test data...")

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Testing rows: {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Identify label
    # --------------------------------------------------------

    if "Label" not in train_df.columns:

        raise ValueError(
            "Label column not found in training data."
        )

    if "Label" not in test_df.columns:

        raise ValueError(
            "Label column not found in test data."
        )

    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X_train = train_df.drop(
        columns=["Label"]
    )

    y_train = train_df["Label"].astype(int)

    X_test = test_df.drop(
        columns=["Label"]
    )

    y_test = test_df["Label"].astype(int)

    # --------------------------------------------------------
    # Verify feature consistency
    # --------------------------------------------------------

    if list(X_train.columns) != list(X_test.columns):

        raise ValueError(
            "Training and testing features do not match."
        )

    feature_names = list(
        X_train.columns
    )

    print(
        f"\nNumber of features: {len(feature_names)}"
    )

    if len(feature_names) != 30:

        raise ValueError(
            f"Expected 30 selected features, "
            f"found {len(feature_names)}."
        )

    # --------------------------------------------------------
    # Convert numeric
    # --------------------------------------------------------

    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    # --------------------------------------------------------
    # NEW SCALER
    # Fit ONLY on training data
    # --------------------------------------------------------

    print("\nFitting new 30-feature StandardScaler...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print(
        "✓ Scaler fitted successfully"
    )

    print(
        f"Scaler feature count: "
        f"{scaler.n_features_in_}"
    )

    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    print(
        "✓ Random Forest training completed"
    )

    print(
        f"Model feature count: "
        f"{model.n_features_in_}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nEvaluating model...")

    y_pred = model.predict(
        X_test_scaled
    )

    y_prob = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "BENIGN",
            "ATTACK"
        ],
        zero_division=0
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BINARY MODEL RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print("\nSaving model...")

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"✓ Saved: {MODEL_FILE}"
    )

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    print(
        f"✓ Saved: {SCALER_FILE}"
    )

    # --------------------------------------------------------
    # Save feature names
    # --------------------------------------------------------

    with open(
        FEATURE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for feature in feature_names:
            f.write(
                feature + "\n"
            )

    print(
        f"✓ Saved: {FEATURE_FILE}"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "BINARY NIDS RANDOM FOREST\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Features : {len(feature_names)}\n"
        )

        f.write(
            f"Accuracy : {accuracy:.6f}\n"
        )

        f.write(
            f"Precision: {precision:.6f}\n"
        )

        f.write(
            f"Recall   : {recall:.6f}\n"
        )

        f.write(
            f"F1 Score : {f1:.6f}\n"
        )

        f.write(
            f"ROC-AUC  : {roc_auc:.6f}\n\n"
        )

        f.write(
            "Classification Report\n"
        )

        f.write(
            "=" * 60 + "\n"
        )

        f.write(report)

    # --------------------------------------------------------
    # Verify artifacts immediately
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VERIFYING MODEL ARTIFACTS")
    print("=" * 70)

    saved_model = joblib.load(
        MODEL_FILE
    )

    saved_scaler = joblib.load(
        SCALER_FILE
    )

    print(
        f"Model expects : "
        f"{saved_model.n_features_in_} features"
    )

    print(
        f"Scaler expects: "
        f"{saved_scaler.n_features_in_} features"
    )

    print(
        f"Current sklearn-compatible artifacts: "
        f"✓"
    )

    if (
        saved_model.n_features_in_ == 30
        and saved_scaler.n_features_in_ == 30
    ):

        print(
            "\n✓ BINARY PIPELINE IS CONSISTENT"
        )

    else:

        print(
            "\n✗ FEATURE MISMATCH STILL EXISTS"
        )

    print("\n" + "=" * 70)
    print("BINARY MODEL RETRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()