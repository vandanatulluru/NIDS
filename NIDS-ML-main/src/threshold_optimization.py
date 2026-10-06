import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "unseen_day_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR, "models", "unseen_day_scaler.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR, "models", "unseen_day_features.txt"
)

RAW_DIR = os.path.join(
    BASE_DIR, "data", "raw"
)

RESULTS_DIR = os.path.join(
    BASE_DIR, "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# FIND WEDNESDAY DATASET
# ============================================================

def find_wednesday_file():

    files = [
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.lower().endswith(".csv")
        and "wednesday" in f.lower()
    ]

    if not files:
        raise FileNotFoundError(
            "Wednesday CSV not found in data/raw/"
        )

    return files[0]


# ============================================================
# LOAD FEATURES
# ============================================================

def load_feature_names():

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        features = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return features


# ============================================================
# CLEAN DATA
# ============================================================

def clean_dataframe(df):

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Replace infinity
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove missing values
    df = df.dropna()

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("THRESHOLD OPTIMIZATION - UNSEEN WEDNESDAY VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features = load_feature_names()

    print(f"Model loaded : {MODEL_PATH}")
    print(f"Scaler loaded: {SCALER_PATH}")
    print(f"Features     : {len(features)}")

    # --------------------------------------------------------
    # Load Wednesday dataset
    # --------------------------------------------------------

    wednesday_file = find_wednesday_file()

    print("\nLoading Wednesday dataset:")
    print(wednesday_file)

    df = pd.read_csv(
        wednesday_file,
        low_memory=False
    )

    print(f"Original rows: {len(df):,}")

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = clean_dataframe(df)

    print(f"Cleaned rows : {len(df):,}")

    # --------------------------------------------------------
    # Check label
    # --------------------------------------------------------

    if "Label" not in df.columns:

        raise ValueError(
            "Label column not found in Wednesday dataset."
        )

    # --------------------------------------------------------
    # Binary labels
    # --------------------------------------------------------

    y = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(
            lambda x: 0
            if x == "BENIGN"
            else 1
        )
        .values
    )

    # --------------------------------------------------------
    # Check features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:

        print("\nMissing features:")

        for feature in missing_features:
            print(" -", feature)

        raise ValueError(
            "Required features are missing."
        )

    # --------------------------------------------------------
    # Prepare X
    # --------------------------------------------------------

    X = df[features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_rows = ~X.isna().any(axis=1)

    X = X.loc[valid_rows]
    y = y[valid_rows]

    X = X.astype(np.float32)

    print(f"\nFinal validation rows: {len(X):,}")

    print("\nClass distribution:")

    print(
        pd.Series(y)
        .map({
            0: "BENIGN",
            1: "ATTACK"
        })
        .value_counts()
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    X_scaled = scaler.transform(X)

    # --------------------------------------------------------
    # Get attack probability
    # --------------------------------------------------------

    print("\nGenerating probabilities...")

    probabilities = model.predict_proba(
        X_scaled
    )[:, 1]

    # ROC-AUC does not depend on threshold
    auc = roc_auc_score(
        y,
        probabilities
    )

    print(f"\nROC-AUC: {auc:.4f}")

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    thresholds = [
        0.50,
        0.45,
        0.40,
        0.35,
        0.30,
        0.25,
        0.20,
        0.15,
        0.10,
        0.05
    ]

    results = []

    print("\n" + "=" * 100)

    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'FPR':<12}"
        f"{'TP':<10}"
        f"{'FP':<10}"
        f"{'FN':<10}"
    )

    print("=" * 100)

    # --------------------------------------------------------
    # Evaluate each threshold
    # --------------------------------------------------------

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y,
            predictions
        )

        precision = precision_score(
            y,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0
        )

        cm = confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) else 0

        results.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp
        })

        print(
            f"{threshold:<12.2f}"
            f"{accuracy:<12.4f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{fpr:<12.4f}"
            f"{tp:<10}"
            f"{fp:<10}"
            f"{fn:<10}"
        )

    print("=" * 100)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    output_csv = os.path.join(
        RESULTS_DIR,
        "threshold_optimization_results.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False
    )

    # --------------------------------------------------------
    # Find best F1 threshold
    # --------------------------------------------------------

    best_f1_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    # --------------------------------------------------------
    # Find best threshold with reasonable precision
    # --------------------------------------------------------

    precision_candidates = results_df[
        results_df["precision"] >= 0.70
    ]

    if len(precision_candidates) > 0:

        best_balanced_row = precision_candidates.loc[
            precision_candidates["f1"].idxmax()
        ]

    else:

        best_balanced_row = best_f1_row

    # --------------------------------------------------------
    # Print recommendation
    # --------------------------------------------------------

    print("\nBEST F1 THRESHOLD")
    print("-" * 40)

    print(
        f"Threshold : "
        f"{best_f1_row['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_f1_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1_row['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{best_f1_row['f1']:.4f}"
    )

    print(
        f"FPR       : "
        f"{best_f1_row['fpr']:.4f}"
    )

    print("\nRECOMMENDED BALANCED THRESHOLD")
    print("-" * 40)

    print(
        f"Threshold : "
        f"{best_balanced_row['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_balanced_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_balanced_row['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{best_balanced_row['f1']:.4f}"
    )

    print(
        f"FPR       : "
        f"{best_balanced_row['fpr']:.4f}"
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = os.path.join(
        RESULTS_DIR,
        "THRESHOLD_OPTIMIZATION_REPORT.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "THRESHOLD OPTIMIZATION REPORT\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            "Dataset: Wednesday CIC-IDS2017\n"
        )

        f.write(
            f"Rows: {len(X):,}\n"
        )

        f.write(
            f"ROC-AUC: {auc:.4f}\n\n"
        )

        f.write(
            "THRESHOLD RESULTS\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(
            results_df.to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "BEST F1 THRESHOLD\n"
        )

        f.write(
            f"{best_f1_row.to_dict()}\n\n"
        )

        f.write(
            "RECOMMENDED BALANCED THRESHOLD\n"
        )

        f.write(
            f"{best_balanced_row.to_dict()}\n"
        )

    print("\nFiles saved:")
    print(output_csv)
    print(report_path)

    print("\n✓ THRESHOLD OPTIMIZATION COMPLETE")


if __name__ == "__main__":
    main()