import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING FILES
#
# Wednesday is deliberately EXCLUDED.
# ============================================================

TRAIN_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv"
]


# ============================================================
# UNSEEN TEST FILE
# ============================================================

TEST_FILE = (
    "Wednesday-workingHours.pcap_ISCX.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "unseen_day_model.pkl"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "unseen_day_scaler.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "unseen_day_features.txt"
)

IMPORTANCE_FILE = os.path.join(
    RESULTS_DIR,
    "unseen_day_feature_importance.csv"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "unseen_day_model_results.csv"
)

REPORT_FILE = os.path.join(
    RESULTS_DIR,
    "UNSEEN_DAY_MODEL_REPORT.txt"
)


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_columns(df):

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(
            " ",
            "_",
            regex=False
        )
    )

    return df


# ============================================================
# LOAD ONE CSV
# ============================================================

def load_file(filename):

    filepath = os.path.join(
        RAW_DIR,
        filename
    )

    print(
        f"\nLoading: {filename}"
    )

    if not os.path.exists(filepath):

        raise FileNotFoundError(
            f"File not found:\n{filepath}"
        )

    df = pd.read_csv(
        filepath,
        low_memory=False
    )

    df = normalize_columns(df)

    print(
        "Rows:",
        len(df)
    )

    if "Label" not in df.columns:

        raise ValueError(
            f"Label column not found in {filename}"
        )

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    # Remove duplicates within each capture
    before = len(df)

    df = df.drop_duplicates()

    print(
        "Duplicates removed:",
        before - len(df)
    )

    return df


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    print("\n")
    print("=" * 70)
    print("LOADING TRAINING CAPTURES")
    print("=" * 70)

    frames = []

    for filename in TRAIN_FILES:

        df = load_file(
            filename
        )

        frames.append(df)

    print(
        "\nCombining training captures..."
    )

    df = pd.concat(
        frames,
        ignore_index=True
    )

    print(
        "Combined training rows:",
        len(df)
    )

    return df


# ============================================================
# LOAD UNSEEN TEST DATA
# ============================================================

def load_test_data():

    print("\n")
    print("=" * 70)
    print("LOADING UNSEEN WEDNESDAY CAPTURE")
    print("=" * 70)

    df = load_file(
        TEST_FILE
    )

    print(
        "\nUnseen test rows:",
        len(df)
    )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    train_df,
    test_df
):

    print("\n")
    print("=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Separate labels
    # --------------------------------------------------------

    y_train = (
        train_df["Label"]
        .str.upper()
        .ne("BENIGN")
        .astype(int)
    )

    y_test = (
        test_df["Label"]
        .str.upper()
        .ne("BENIGN")
        .astype(int)
    )

    # --------------------------------------------------------
    # Keep only numeric columns
    # --------------------------------------------------------

    train_features = train_df.drop(
        columns=["Label"]
    )

    test_features = test_df.drop(
        columns=["Label"]
    )

    # Find common columns
    common_columns = [
        col
        for col in train_features.columns
        if col in test_features.columns
    ]

    train_features = train_features[
        common_columns
    ]

    test_features = test_features[
        common_columns
    ]

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    train_features = train_features.apply(
        pd.to_numeric,
        errors="coerce"
    )

    test_features = test_features.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Replace invalid values
    # --------------------------------------------------------

    train_features = (
        train_features
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    test_features = (
        test_features
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    print(
        "Initial feature count:",
        train_features.shape[1]
    )

    return (
        train_features,
        test_features,
        y_train,
        y_test
    )


# ============================================================
# FEATURE SELECTION
#
# IMPORTANT:
# Feature selection is fitted ONLY on training data.
# ============================================================

def select_features(
    X_train,
    X_test
):

    print("\n")
    print("=" * 70)
    print("TRAINING-ONLY FEATURE SELECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Remove zero-variance features
    # --------------------------------------------------------

    variance_selector = VarianceThreshold(
        threshold=0.0
    )

    X_train_var = variance_selector.fit_transform(
        X_train
    )

    X_test_var = variance_selector.transform(
        X_test
    )

    variance_features = X_train.columns[
        variance_selector.get_support()
    ]

    X_train_var = pd.DataFrame(
        X_train_var,
        columns=variance_features,
        index=X_train.index
    )

    X_test_var = pd.DataFrame(
        X_test_var,
        columns=variance_features,
        index=X_test.index
    )

    print(
        "After variance filtering:",
        X_train_var.shape[1]
    )

    # --------------------------------------------------------
    # Random Forest feature importance
    #
    # Use a stratified sample for computational efficiency.
    # --------------------------------------------------------

    max_sample = min(
        150000,
        len(X_train_var)
    )

    rng = np.random.RandomState(
        42
    )

    # Stratified sample
    y_temp = np.asarray(
        TRAIN_Y_FOR_SELECTION
    )

    benign_indices = np.where(
        y_temp == 0
    )[0]

    attack_indices = np.where(
        y_temp == 1
    )[0]

    # Half sample from each class where possible
    benign_count = min(
        max_sample // 2,
        len(benign_indices)
    )

    attack_count = min(
        max_sample - benign_count,
        len(attack_indices)
    )

    selected_indices = np.concatenate([
        rng.choice(
            benign_indices,
            benign_count,
            replace=False
        ),
        rng.choice(
            attack_indices,
            attack_count,
            replace=False
        )
    ])

    rng.shuffle(
        selected_indices
    )

    X_sample = X_train_var.iloc[
        selected_indices
    ]

    y_sample = y_temp[
        selected_indices
    ]

    print(
        "Feature-selection sample:",
        len(X_sample)
    )

    selector_model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    selector_model.fit(
        X_sample,
        y_sample
    )

    importance = pd.Series(
        selector_model.feature_importances_,
        index=X_train_var.columns
    ).sort_values(
        ascending=False
    )

    # --------------------------------------------------------
    # Select top 30
    # --------------------------------------------------------

    selected_features = (
        importance
        .head(30)
        .index
        .tolist()
    )

    print(
        "\nSelected 30 features:"
    )

    for i, feature in enumerate(
        selected_features,
        start=1
    ):

        print(
            f"{i:2d}. {feature}"
        )

    # --------------------------------------------------------
    # Save feature importance
    # --------------------------------------------------------

    importance_df = pd.DataFrame({

        "Feature":
            importance.index,

        "Importance":
            importance.values

    })

    importance_df.to_csv(
        IMPORTANCE_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Return selected data
    # --------------------------------------------------------

    X_train_selected = (
        X_train_var[
            selected_features
        ]
    )

    X_test_selected = (
        X_test_var[
            selected_features
        ]
    )

    return (
        X_train_selected,
        X_test_selected,
        selected_features
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):

    print("\n")
    print("=" * 70)
    print("TRAINING UNSEEN-DAY MODEL")
    print("=" * 70)

    print(
        "\nTraining rows:",
        len(X_train)
    )

    print(
        "Features:",
        X_train.shape[1]
    )

    print(
        "Benign:",
        int((y_train == 0).sum())
    )

    print(
        "Attack:",
        int((y_train == 1).sum())
    )

    # --------------------------------------------------------
    # Scale ONLY training data
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print(
        "\nTraining Random Forest..."
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    print(
        "✓ Training completed"
    )

    return model, scaler


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    scaler,
    X_test,
    y_test
):

    print("\n")
    print("=" * 70)
    print("UNSEEN WEDNESDAY MODEL EVALUATION")
    print("=" * 70)

    X_test_scaled = scaler.transform(
        X_test
    )

    predictions = model.predict(
        X_test_scaled
    )

    probabilities = (
        model.predict_proba(
            X_test_scaled
        )[:, 1]
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
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

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "BENIGN",
                "ATTACK"
            ],
            zero_division=0
        )
    )

    return (
        predictions,
        probabilities,
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm
        }
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    scaler,
    features
):

    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    with open(
        FEATURE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for feature in features:

            f.write(
                feature + "\n"
            )

    print(
        "\nModel saved:",
        MODEL_FILE
    )

    print(
        "Scaler saved:",
        SCALER_FILE
    )

    print(
        "Features saved:",
        FEATURE_FILE
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    y_original,
    y_test,
    predictions,
    probabilities
):

    results = pd.DataFrame({

        "Ground_Truth_Label":
            y_original.values,

        "Ground_Truth_Binary":
            np.where(
                y_test.values == 1,
                "ATTACK",
                "BENIGN"
            ),

        "Prediction":
            np.where(
                predictions == 1,
                "ATTACK",
                "BENIGN"
            ),

        "Attack_Probability":
            probabilities

    })

    results["Correct"] = (
        results["Ground_Truth_Binary"]
        ==
        results["Prediction"]
    )

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        "\nDetailed results saved:",
        RESULTS_FILE
    )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    metrics,
    train_rows,
    test_rows,
    selected_features
):

    lines = []

    lines.append(
        "=" * 70
    )

    lines.append(
        "UNSEEN-DAY MODEL TRAINING AND VALIDATION REPORT"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        "\nTraining strategy:"
    )

    lines.append(
        "Monday + Tuesday + Thursday + Friday"
    )

    lines.append(
        "\nUnseen test:"
    )

    lines.append(
        "Wednesday-workingHours.pcap_ISCX.csv"
    )

    lines.append(
        f"\nTraining rows: {train_rows}"
    )

    lines.append(
        f"Unseen test rows: {test_rows}"
    )

    lines.append(
        "\nNumber of selected features:"
    )

    lines.append(
        str(len(selected_features))
    )

    lines.append(
        "\nSelected features:"
    )

    for feature in selected_features:

        lines.append(
            f"- {feature}"
        )

    lines.append(
        "\nBinary Detection Performance:"
    )

    lines.append(
        "-" * 50
    )

    lines.append(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    lines.append(
        f"Precision: {metrics['precision']:.4f}"
    )

    lines.append(
        f"Recall   : {metrics['recall']:.4f}"
    )

    lines.append(
        f"F1 Score : {metrics['f1']:.4f}"
    )

    lines.append(
        f"ROC-AUC  : {metrics['roc_auc']:.4f}"
    )

    lines.append(
        "\nConfusion Matrix:"
    )

    lines.append(
        str(metrics["confusion_matrix"])
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )

    print(
        "\nReport saved:",
        REPORT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global TRAIN_Y_FOR_SELECTION

    print("=" * 70)
    print("TRAINING DAY-AWARE NIDS MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Load training captures
    # --------------------------------------------------------

    train_df = load_training_data()

    # --------------------------------------------------------
    # Load completely unseen Wednesday capture
    # --------------------------------------------------------

    test_df = load_test_data()

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_features(
        train_df,
        test_df
    )

    # Global used only by training-only feature selection
    TRAIN_Y_FOR_SELECTION = y_train.values

    # --------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------

    (
        X_train_selected,
        X_test_selected,
        selected_features
    ) = select_features(
        X_train,
        X_test
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model, scaler = train_model(
        X_train_selected,
        y_train
    )

    # --------------------------------------------------------
    # Evaluate unseen Wednesday
    # --------------------------------------------------------

    (
        predictions,
        probabilities,
        metrics
    ) = evaluate_model(
        model,
        scaler,
        X_test_selected,
        y_test
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        model,
        scaler,
        selected_features
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    save_results(
        test_df["Label"],
        y_test,
        predictions,
        probabilities
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    save_report(
        metrics,
        len(X_train_selected),
        len(X_test_selected),
        selected_features
    )

    print("\n")
    print("=" * 70)
    print("DAY-AWARE TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()