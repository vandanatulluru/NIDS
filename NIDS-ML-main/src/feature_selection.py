"""
Feature Selection Module for NIDS-ML

Leakage-safe feature selection pipeline.

The feature-selection rules are learned ONLY from the training data.
The same selected features are then applied to the test data.

Methods:
1. Variance filtering
2. Random Forest feature importance
3. Top-N feature selection
4. Save selected train/test datasets
5. Save selected feature names
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
LOG_DIR = PROJECT_ROOT / "logs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "feature_selection.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Number of final features
N_SELECTED_FEATURES = 30

# Random Forest used only for feature ranking
RF_ESTIMATORS = 100

# Maximum samples used to learn feature importance
# This keeps computation manageable for CIC-IDS2017.
MAX_SAMPLES_FOR_SELECTION = 100_000


# ============================================================
# STEP 1 — LOAD TRAIN / TEST DATA
# ============================================================

def load_train_test_data():

    logger.info("=" * 70)
    logger.info("STEP 1: LOADING PREPROCESSED TRAIN / TEST DATA")
    logger.info("=" * 70)

    train_path = (
        PROCESSED_DIR /
        "train_data.csv"
    )

    test_path = (
        PROCESSED_DIR /
        "test_data.csv"
    )

    if not train_path.exists():

        raise FileNotFoundError(
            f"Training data not found:\n{train_path}\n\n"
            "Run data_preprocessing.py first."
        )

    if not test_path.exists():

        raise FileNotFoundError(
            f"Testing data not found:\n{test_path}\n\n"
            "Run data_preprocessing.py first."
        )

    logger.info(
        f"Loading training data: {train_path}"
    )

    train_df = pd.read_csv(
        train_path
    )

    logger.info(
        f"Training shape: {train_df.shape}"
    )

    logger.info(
        f"Loading testing data: {test_path}"
    )

    test_df = pd.read_csv(
        test_path
    )

    logger.info(
        f"Testing shape: {test_df.shape}"
    )

    # --------------------------------------------------------
    # Identify label
    # --------------------------------------------------------

    label_candidates = [
        "Label",
        "label",
        "Attack",
        "attack",
        "Class",
        "class"
    ]

    label_col = None

    for candidate in label_candidates:

        if candidate in train_df.columns:

            label_col = candidate
            break

    if label_col is None:

        raise ValueError(
            "Label column not found."
        )

    if label_col not in test_df.columns:

        raise ValueError(
            "Label column missing from test dataset."
        )

    logger.info(
        f"Label column: {label_col}"
    )

    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X_train = train_df.drop(
        columns=[label_col]
    )

    y_train = train_df[label_col]

    X_test = test_df.drop(
        columns=[label_col]
    )

    y_test = test_df[label_col]

    # --------------------------------------------------------
    # Ensure same feature columns
    # --------------------------------------------------------

    if list(X_train.columns) != list(X_test.columns):

        raise ValueError(
            "Training and testing feature columns do not match."
        )

    logger.info(
        f"Number of initial features: "
        f"{X_train.shape[1]}"
    )

    logger.info(
        f"Training samples: "
        f"{len(X_train):,}"
    )

    logger.info(
        f"Testing samples: "
        f"{len(X_test):,}"
    )

    logger.info("=" * 70)

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# STEP 2 — VARIANCE FILTERING
# ============================================================

def variance_filter(
    X_train,
    X_test
):

    logger.info("=" * 70)
    logger.info("STEP 2: VARIANCE FILTERING")
    logger.info("=" * 70)

    initial_features = X_train.shape[1]

    # Fit selector ONLY on training data
    selector = VarianceThreshold(
        threshold=0.0
    )

    selector.fit(X_train)

    selected_mask = selector.get_support()

    selected_features = (
        X_train.columns[selected_mask]
        .tolist()
    )

    removed_features = [
        feature
        for feature in X_train.columns
        if feature not in selected_features
    ]

    X_train_filtered = X_train[
        selected_features
    ].copy()

    X_test_filtered = X_test[
        selected_features
    ].copy()

    logger.info(
        f"Initial features: "
        f"{initial_features}"
    )

    logger.info(
        f"Remaining features: "
        f"{len(selected_features)}"
    )

    logger.info(
        f"Removed constant features: "
        f"{len(removed_features)}"
    )

    if removed_features:

        logger.info(
            "Removed features:"
        )

        for feature in removed_features:

            logger.info(
                f"  - {feature}"
            )

    logger.info("=" * 70)

    return (
        X_train_filtered,
        X_test_filtered
    )


# ============================================================
# STEP 3 — RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

def calculate_feature_importance(
    X_train,
    y_train
):

    logger.info("=" * 70)
    logger.info(
        "STEP 3: RANDOM FOREST FEATURE IMPORTANCE"
    )
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Sample large training dataset
    # --------------------------------------------------------

    if len(X_train) > MAX_SAMPLES_FOR_SELECTION:

        logger.info(
            f"Training data contains "
            f"{len(X_train):,} samples."
        )

        logger.info(
            f"Using stratified sample of "
            f"{MAX_SAMPLES_FOR_SELECTION:,} samples "
            f"for feature ranking."
        )

        from sklearn.model_selection import train_test_split

        X_sample, _, y_sample, _ = train_test_split(
            X_train,
            y_train,
            train_size=MAX_SAMPLES_FOR_SELECTION,
            random_state=RANDOM_STATE,
            stratify=y_train
        )

    else:

        X_sample = X_train
        y_sample = y_train

    logger.info(
        f"Feature-ranking sample: "
        f"{len(X_sample):,} rows"
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    rf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        verbose=0
    )

    logger.info(
        "Training Random Forest for "
        "feature importance..."
    )

    rf.fit(
        X_sample,
        y_sample
    )

    logger.info(
        "Random Forest feature ranking complete."
    )

    # --------------------------------------------------------
    # Importance dataframe
    # --------------------------------------------------------

    importance_df = pd.DataFrame({

        "Feature": X_train.columns,

        "Importance": rf.feature_importances_

    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save importance
    # --------------------------------------------------------

    importance_path = (
        RESULTS_DIR /
        "feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    logger.info(
        f"Feature importance saved to:"
    )

    logger.info(
        f"  {importance_path}"
    )

    # --------------------------------------------------------
    # Display top features
    # --------------------------------------------------------

    logger.info(
        f"\nTop {N_SELECTED_FEATURES} features:"
    )

    for index, row in importance_df.head(
        N_SELECTED_FEATURES
    ).iterrows():

        logger.info(
            f"  {index + 1}. "
            f"{row['Feature']} = "
            f"{row['Importance']:.6f}"
        )

    logger.info("=" * 70)

    return importance_df


# ============================================================
# STEP 4 — SELECT TOP FEATURES
# ============================================================

def select_top_features(
    X_train,
    X_test,
    importance_df
):

    logger.info("=" * 70)
    logger.info("STEP 4: SELECTING TOP FEATURES")
    logger.info("=" * 70)

    available_features = set(
        X_train.columns
    )

    selected_features = [
        feature
        for feature in importance_df["Feature"].tolist()
        if feature in available_features
    ][:N_SELECTED_FEATURES]

    if len(selected_features) == 0:

        raise ValueError(
            "No features were selected."
        )

    X_train_selected = X_train[
        selected_features
    ].copy()

    X_test_selected = X_test[
        selected_features
    ].copy()

    logger.info(
        f"Selected {len(selected_features)} "
        f"features from "
        f"{X_train.shape[1]}"
    )

    logger.info(
        "\nSelected features:"
    )

    for index, feature in enumerate(
        selected_features,
        start=1
    ):

        logger.info(
            f"  {index}. {feature}"
        )

    logger.info("=" * 70)

    return (
        X_train_selected,
        X_test_selected,
        selected_features
    )


# ============================================================
# STEP 5 — SAVE SELECTED FEATURES
# ============================================================

def save_selected_data(
    X_train,
    X_test,
    y_train,
    y_test,
    selected_features
):

    logger.info("=" * 70)
    logger.info(
        "STEP 5: SAVING SELECTED FEATURE DATA"
    )
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_selected = X_train.copy()

    train_selected["Label"] = (
        y_train.values
    )

    train_path = (
        PROCESSED_DIR /
        "selected_train_data.csv"
    )

    train_selected.to_csv(
        train_path,
        index=False
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_selected = X_test.copy()

    test_selected["Label"] = (
        y_test.values
    )

    test_path = (
        PROCESSED_DIR /
        "selected_test_data.csv"
    )

    test_selected.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    features_path = (
        RESULTS_DIR /
        "selected_features.txt"
    )

    with open(
        features_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "NIDS-ML Selected Features\n"
        )

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            f"Total Selected Features: "
            f"{len(selected_features)}\n"
        )

        file.write(
            "Selection Method: "
            "Random Forest Feature Importance\n"
        )

        file.write(
            "Selection fitted ONLY on training data\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        for index, feature in enumerate(
            selected_features,
            start=1
        ):

            file.write(
                f"{index}. {feature}\n"
            )

    logger.info(
        f"Selected training data saved:"
    )

    logger.info(
        f"  {train_path}"
    )

    logger.info(
        f"Selected testing data saved:"
    )

    logger.info(
        f"  {test_path}"
    )

    logger.info(
        f"Selected feature list saved:"
    )

    logger.info(
        f"  {features_path}"
    )

    logger.info("=" * 70)

    return (
        train_path,
        test_path,
        features_path
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "NIDS-ML LEAKAGE-SAFE FEATURE SELECTION"
    )
    logger.info("=" * 70)
    logger.info("")

    try:

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = load_train_test_data()

        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------

        (
            X_train_filtered,
            X_test_filtered
        ) = variance_filter(
            X_train,
            X_test
        )

        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------

        importance_df = calculate_feature_importance(
            X_train_filtered,
            y_train
        )

        # ----------------------------------------------------
        # STEP 4
        # ----------------------------------------------------

        (
            X_train_selected,
            X_test_selected,
            selected_features
        ) = select_top_features(
            X_train_filtered,
            X_test_filtered,
            importance_df
        )

        # ----------------------------------------------------
        # STEP 5
        # ----------------------------------------------------

        save_selected_data(
            X_train_selected,
            X_test_selected,
            y_train,
            y_test,
            selected_features
        )

        # ----------------------------------------------------
        # FINAL SUMMARY
        # ----------------------------------------------------

        logger.info("")
        logger.info("=" * 70)
        logger.info(
            "FEATURE SELECTION COMPLETED SUCCESSFULLY!"
        )
        logger.info("=" * 70)

        logger.info(
            f"Initial features: "
            f"{X_train.shape[1]}"
        )

        logger.info(
            f"After variance filtering: "
            f"{X_train_filtered.shape[1]}"
        )

        logger.info(
            f"Final selected features: "
            f"{len(selected_features)}"
        )

        logger.info(
            f"Feature reduction: "
            f"{X_train.shape[1] - len(selected_features)}"
        )

        logger.info(
            "\nGenerated files:"
        )

        logger.info(
            "  data/processed/selected_train_data.csv"
        )

        logger.info(
            "  data/processed/selected_test_data.csv"
        )

        logger.info(
            "  results/feature_importance.csv"
        )

        logger.info(
            "  results/selected_features.txt"
        )

        logger.info("=" * 70)

    except Exception as e:

        logger.error(
            "Feature selection failed.",
            exc_info=True
        )

        raise


if __name__ == "__main__":
    main()