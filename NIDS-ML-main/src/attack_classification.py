import os
import json
import glob
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import joblib

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
PLOT_DIR = os.path.join(RESULT_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

RANDOM_STATE = 42

# Use the same 30 features selected by the binary model
FEATURE_FILE = os.path.join(MODEL_DIR, "feature_names.txt")

MODEL_FILE = os.path.join(MODEL_DIR, "attack_classifier.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "multiclass_scaler.pkl")
ENCODER_FILE = os.path.join(MODEL_DIR, "attack_label_encoder.pkl")
CLASSES_FILE = os.path.join(MODEL_DIR, "attack_classes.json")

METRICS_FILE = os.path.join(RESULT_DIR, "attack_classification_metrics.txt")
CM_FILE = os.path.join(RESULT_DIR, "attack_confusion_matrix.csv")


# ============================================================
# LOAD SELECTED FEATURES
# ============================================================

def load_selected_features():

    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        features = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(f"Selected features loaded: {len(features)}")

    return features


# ============================================================
# LOAD RAW CIC-IDS2017 DATA
# ============================================================

def load_raw_dataset():

    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}"
        )

    print(f"\nFound {len(files)} CSV files")

    dataframes = []

    for file in files:

        print(f"Loading: {os.path.basename(file)}")

        df = pd.read_csv(
            file,
            low_memory=False
        )

        # Remove spaces around column names
        df.columns = df.columns.str.strip()

        dataframes.append(df)

    data = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(f"\nTotal rows: {len(data):,}")

    return data


# ============================================================
# CLEAN DATA
# ============================================================

def clean_dataset(df):

    print("\nCleaning dataset...")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Replace infinity
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Label column
    if "Label" not in df.columns:
        raise ValueError(
            "Label column not found."
        )

    # Remove rows without labels
    df = df.dropna(subset=["Label"])

    # Clean label text
    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    # Remove leading/trailing spaces from every column
    df.columns = df.columns.str.strip()

    print(f"Rows after cleaning: {len(df):,}")

    print("\nAttack distribution:")
    print(
        df["Label"]
        .value_counts()
        .to_string()
    )

    return df


# ============================================================
# NORMALIZE FEATURE NAMES
# ============================================================

def normalize_feature_names(df):

    mapping = {}

    for col in df.columns:

        normalized = (
            col.strip()
            .replace(" ", "_")
        )

        mapping[col] = normalized

    df = df.rename(
        columns=mapping
    )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df, selected_features):

    df = normalize_feature_names(df)

    # Normalize selected feature names
    selected_features = [
        f.strip().replace(" ", "_")
        for f in selected_features
    ]

    missing = [
        f for f in selected_features
        if f not in df.columns
    ]

    if missing:

        print("\nWARNING: Missing features:")

        for feature in missing:
            print(" -", feature)

        raise ValueError(
            "Some selected features are missing "
            "from the raw CIC-IDS2017 dataset."
        )

    X = df[selected_features].copy()

    # Convert everything to numeric
    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Replace invalid values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values with training-independent
    # neutral value. The scaler is fitted later.
    X = X.fillna(0)

    return X


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_multiclass_classifier():

    print("=" * 70)
    print("CIC-IDS2017 MULTICLASS ATTACK CLASSIFIER")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load selected features
    # --------------------------------------------------------

    selected_features = load_selected_features()

    # --------------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------------

    df = load_raw_dataset()

    # --------------------------------------------------------
    # 3. Clean
    # --------------------------------------------------------

    df = clean_dataset(df)

    # --------------------------------------------------------
    # 4. Prepare X
    # --------------------------------------------------------

    X = prepare_features(
        df,
        selected_features
    )

    y = df["Label"].copy()

    print(
        f"\nFeature matrix: "
        f"{X.shape[0]:,} rows × {X.shape[1]} features"
    )

    # --------------------------------------------------------
    # 5. Remove classes with too few samples
    #
    # The dataset contains extremely rare classes:
    # Infiltration = 36
    # SQL Injection = 21
    # Heartbleed = 11
    #
    # A stratified 80/20 split requires at least 2 samples.
    # We keep every class because all have >= 2 samples.
    # --------------------------------------------------------

    print("\nClasses:")
    print(y.value_counts().to_string())

    # --------------------------------------------------------
    # 6. Train/Test split FIRST
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain samples:", f"{len(X_train):,}")
    print("Test samples :", f"{len(X_test):,}")

    # --------------------------------------------------------
    # 7. Encode labels
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(
        y_train
    )

    y_test_encoded = label_encoder.transform(
        y_test
    )

    classes = list(
        label_encoder.classes_
    )

    print("\nEncoded classes:")

    for i, class_name in enumerate(classes):
        print(f"{i}: {class_name}")

    # --------------------------------------------------------
    # 8. Scale using TRAINING DATA ONLY
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # 9. Train Random Forest
    #
    # class_weight='balanced' helps compensate for
    # the extreme class imbalance.
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train_scaled,
        y_train_encoded
    )

    print("Training completed.")

    # --------------------------------------------------------
    # 10. Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(
        X_test_scaled
    )

    # --------------------------------------------------------
    # 11. Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test_encoded,
        y_pred
    )

    precision = precision_score(
        y_test_encoded,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test_encoded,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test_encoded,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("MULTICLASS RESULTS")
    print("=" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # --------------------------------------------------------
    # 12. Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test_encoded,
        y_pred,
        target_names=classes,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # 13. Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test_encoded,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    cm_df.to_csv(
        CM_FILE
    )

    # --------------------------------------------------------
    # 14. Save metrics
    # --------------------------------------------------------

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CIC-IDS2017 MULTICLASS ATTACK CLASSIFIER\n"
        )

        f.write("=" * 60 + "\n\n")

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
            f"F1 Score : {f1:.6f}\n\n"
        )

        f.write(
            "Classification Report\n"
        )

        f.write("=" * 60 + "\n")

        f.write(report)

    # --------------------------------------------------------
    # 15. Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    joblib.dump(
        label_encoder,
        ENCODER_FILE
    )

    # Save class names as JSON
    with open(
        CLASSES_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            classes,
            f,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # 16. Save feature names
    # --------------------------------------------------------

    with open(
        os.path.join(
            MODEL_DIR,
            "attack_feature_names.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        for feature in selected_features:
            f.write(
                feature + "\n"
            )

    # --------------------------------------------------------
    # 17. Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MULTICLASS TRAINING COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        "  models/attack_classifier.pkl"
    )

    print(
        "  models/multiclass_scaler.pkl"
    )

    print(
        "  models/attack_label_encoder.pkl"
    )

    print(
        "  models/attack_classes.json"
    )

    print(
        "  models/attack_feature_names.txt"
    )

    print(
        "  results/attack_classification_metrics.txt"
    )

    print(
        "  results/attack_confusion_matrix.csv"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train_multiclass_classifier()