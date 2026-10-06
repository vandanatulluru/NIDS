"""
Data Preprocessing Module for NIDS-ML

Corrected pipeline:

1. Load CIC-IDS2017 CSV files
2. Clean column names
3. Handle missing/infinite values
4. Remove duplicate rows
5. Convert labels to binary:
      BENIGN = 0
      ATTACK = 1
6. Split into train/test using stratification
7. Fit StandardScaler ONLY on training data
8. Transform train and test data
9. Optionally apply SMOTE ONLY to training data
10. Save train/test datasets separately

This prevents test-data leakage during preprocessing.
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


# ============================================================
# PATHS AND LOGGING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "preprocessing.log",
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
TEST_SIZE = 0.20

# IMPORTANT:
# Keep this False initially because CIC-IDS2017 is very large.
# We can enable SMOTE later if required.
APPLY_SMOTE = False

# When SMOTE is enabled, attack samples will be increased
# to 50% of the benign class.
SMOTE_RATIO = 0.50


# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================

def load_dataset(data_dir=RAW_DATA_DIR):
    """
    Load all CSV files directly from data/raw and combine them.
    """

    logger.info("=" * 70)
    logger.info("STEP 1: LOADING CIC-IDS2017 DATASET")
    logger.info("=" * 70)

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {data_path}"
        )

    csv_files = sorted(data_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {data_path}"
        )

    logger.info(
        f"Found {len(csv_files)} CSV file(s)"
    )

    dataframes = []

    for csv_file in csv_files:

        logger.info(f"Loading: {csv_file.name}")

        try:

            df = pd.read_csv(
                csv_file,
                encoding="utf-8",
                low_memory=False
            )

            logger.info(
                f"  Loaded {len(df):,} rows "
                f"x {len(df.columns)} columns"
            )

            dataframes.append(df)

        except Exception as e:

            logger.error(
                f"Failed to load {csv_file.name}: {e}"
            )
            raise

    if not dataframes:
        raise ValueError(
            "No CSV data could be loaded."
        )

    logger.info("Combining CSV files...")

    df = pd.concat(
        dataframes,
        ignore_index=True
    )

    logger.info(
        f"Combined dataset: "
        f"{len(df):,} rows x {len(df.columns)} columns"
    )

    logger.info(
        f"Memory usage: "
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )

    logger.info("=" * 70)

    return df


# ============================================================
# STEP 2 — CLEAN DATA
# ============================================================

def clean_dataset(df):
    """
    Clean CIC-IDS2017 data.

    Operations:
    - Clean column names
    - Remove duplicate rows
    - Replace infinite values
    - Handle missing values
    """

    logger.info("=" * 70)
    logger.info("STEP 2: CLEANING DATASET")
    logger.info("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )

    logger.info("Column names cleaned.")

    # --------------------------------------------------------
    # Find label column
    # --------------------------------------------------------

    label_candidates = [
        "Label",
        "label",
        "Class",
        "class"
    ]

    label_col = None

    for candidate in label_candidates:

        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:

        possible_labels = [
            col for col in df.columns
            if "label" in col.lower()
            or "class" in col.lower()
        ]

        if possible_labels:
            label_col = possible_labels[0]

    if label_col is None:

        raise ValueError(
            "Could not find Label/Class column."
        )

    logger.info(
        f"Label column detected: {label_col}"
    )

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:

        logger.info(
            f"Removing {duplicate_count:,} duplicate rows..."
        )

        df = df.drop_duplicates()

    else:

        logger.info("No duplicate rows found.")

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    infinite_count = 0

    for column in numeric_columns:

        inf_mask = np.isinf(df[column])

        if inf_mask.any():

            count = int(inf_mask.sum())
            infinite_count += count

            df.loc[
                inf_mask & (df[column] > 0),
                column
            ] = np.nan

            df.loc[
                inf_mask & (df[column] < 0),
                column
            ] = np.nan

    logger.info(
        f"Infinite values converted to NaN: "
        f"{infinite_count:,}"
    )

    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in numeric_columns:

        if df[column].isnull().any():

            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )

    for column in categorical_columns:

        if df[column].isnull().any():

            mode = df[column].mode()

            if len(mode) > 0:
                replacement = mode.iloc[0]
            else:
                replacement = "Unknown"

            df[column] = df[column].fillna(
                replacement
            )

    logger.info(
        f"Dataset after cleaning: "
        f"{len(df):,} rows x {len(df.columns)} columns"
    )

    logger.info("=" * 70)

    return df, label_col


# ============================================================
# STEP 3 — ENCODE LABEL
# ============================================================

def prepare_features_and_labels(df, label_col):
    """
    Convert CIC-IDS2017 labels into binary classification.

    BENIGN -> 0
    Everything else -> 1
    """

    logger.info("=" * 70)
    logger.info("STEP 3: PREPARING FEATURES AND LABELS")
    logger.info("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Display original labels
    # --------------------------------------------------------

    logger.info("Original label distribution:")

    label_counts = df[label_col].value_counts()

    for label, count in label_counts.items():

        logger.info(
            f"  {label}: {count:,}"
        )

    # --------------------------------------------------------
    # Binary labels
    # --------------------------------------------------------

    df["Binary_Label"] = (
        df[label_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(
            lambda x: 0 if x == "BENIGN" else 1
        )
    )

    y = df["Binary_Label"].astype(np.int8)

    # --------------------------------------------------------
    # Remove label columns
    # --------------------------------------------------------

    X = df.drop(
        columns=[
            label_col,
            "Binary_Label"
        ],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Handle categorical features
    # --------------------------------------------------------

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    if categorical_columns:

        logger.info(
            f"Encoding {len(categorical_columns)} "
            f"categorical feature(s)"
        )

        for column in categorical_columns:

            # CIC-IDS2017 is primarily numeric.
            # Any remaining categorical feature is
            # converted into category codes.

            X[column] = (
                X[column]
                .astype("category")
                .cat.codes
            )

    # --------------------------------------------------------
    # Convert all features to numeric
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace any conversion-generated NaNs
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    # --------------------------------------------------------
    # Remove columns with no usable values
    # --------------------------------------------------------

    invalid_columns = [
        column
        for column in X.columns
        if X[column].isna().all()
    ]

    if invalid_columns:

        logger.warning(
            f"Dropping {len(invalid_columns)} "
            f"empty feature columns."
        )

        X = X.drop(
            columns=invalid_columns
        )

    # --------------------------------------------------------
    # Final class distribution
    # --------------------------------------------------------

    benign_count = int(
        (y == 0).sum()
    )

    attack_count = int(
        (y == 1).sum()
    )

    logger.info("")
    logger.info("Binary label distribution:")

    logger.info(
        f"  BENIGN (0): "
        f"{benign_count:,} "
        f"({benign_count / len(y) * 100:.2f}%)"
    )

    logger.info(
        f"  ATTACK (1): "
        f"{attack_count:,} "
        f"({attack_count / len(y) * 100:.2f}%)"
    )

    logger.info("")
    logger.info(
        f"Feature matrix: "
        f"{X.shape}"
    )

    logger.info(
        f"Label vector: "
        f"{y.shape}"
    )

    logger.info("=" * 70)

    return X, y


# ============================================================
# STEP 4 — TRAIN / TEST SPLIT
# ============================================================

def split_dataset(X, y):
    """
    Split the dataset BEFORE fitting the scaler or SMOTE.

    This is the key leakage-prevention step.
    """

    logger.info("=" * 70)
    logger.info("STEP 4: TRAIN / TEST SPLIT")
    logger.info("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.info(
        f"Training samples: "
        f"{len(X_train):,}"
    )

    logger.info(
        f"Testing samples: "
        f"{len(X_test):,}"
    )

    logger.info(
        f"Training class distribution:"
    )

    logger.info(
        f"  BENIGN: {(y_train == 0).sum():,}"
    )

    logger.info(
        f"  ATTACK: {(y_train == 1).sum():,}"
    )

    logger.info(
        f"Testing class distribution:"
    )

    logger.info(
        f"  BENIGN: {(y_test == 0).sum():,}"
    )

    logger.info(
        f"  ATTACK: {(y_test == 1).sum():,}"
    )

    logger.info("=" * 70)

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# STEP 5 — SCALE FEATURES
# ============================================================

def scale_features(
    X_train,
    X_test
):
    """
    Fit StandardScaler ONLY on training data.

    Test data is transformed using the scaler
    learned from training data.
    """

    logger.info("=" * 70)
    logger.info("STEP 5: FEATURE SCALING")
    logger.info("=" * 70)

    scaler = StandardScaler()

    # IMPORTANT:
    # fit_transform ONLY on training data
    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # Test data is ONLY transformed
    X_test_scaled = scaler.transform(
        X_test
    )

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    logger.info(
        "Scaler fitted only on training data."
    )

    logger.info(
        f"Scaled training shape: "
        f"{X_train_scaled.shape}"
    )

    logger.info(
        f"Scaled testing shape: "
        f"{X_test_scaled.shape}"
    )

    logger.info("=" * 70)

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )


# ============================================================
# STEP 6 — OPTIONAL SMOTE
# ============================================================

def apply_smote_to_training_data(
    X_train,
    y_train
):
    """
    Apply SMOTE ONLY to the training set.

    Test data is never modified.
    """

    logger.info("=" * 70)
    logger.info("STEP 6: SMOTE CLASS BALANCING")
    logger.info("=" * 70)

    if not APPLY_SMOTE:

        logger.info(
            "SMOTE is DISABLED."
        )

        logger.info(
            "Training data will remain unchanged."
        )

        logger.info("=" * 70)

        return X_train, y_train

    benign_count = int(
        (y_train == 0).sum()
    )

    attack_count = int(
        (y_train == 1).sum()
    )

    logger.info(
        f"Before SMOTE:"
    )

    logger.info(
        f"  BENIGN: {benign_count:,}"
    )

    logger.info(
        f"  ATTACK: {attack_count:,}"
    )

    # Only oversample when attacks are the minority class
    if attack_count >= benign_count:

        logger.info(
            "Attack class is not a minority class."
        )

        logger.info(
            "Skipping SMOTE."
        )

        return X_train, y_train

    target_attack_count = int(
        benign_count * SMOTE_RATIO
    )

    if target_attack_count <= attack_count:

        logger.info(
            "Target attack count is not larger "
            "than current attack count."
        )

        return X_train, y_train

    logger.info(
        f"SMOTE target attack samples: "
        f"{target_attack_count:,}"
    )

    smote = SMOTE(
        sampling_strategy={
            1: target_attack_count
        },
        random_state=RANDOM_STATE,
        k_neighbors=5
    )

    X_balanced, y_balanced = smote.fit_resample(
        X_train,
        y_train
    )

    X_balanced = pd.DataFrame(
        X_balanced,
        columns=X_train.columns
    )

    y_balanced = pd.Series(
        y_balanced,
        name="Label"
    )

    logger.info(
        "After SMOTE:"
    )

    logger.info(
        f"  BENIGN: "
        f"{(y_balanced == 0).sum():,}"
    )

    logger.info(
        f"  ATTACK: "
        f"{(y_balanced == 1).sum():,}"
    )

    logger.info(
        f"Total training samples: "
        f"{len(y_balanced):,}"
    )

    logger.info("=" * 70)

    return X_balanced, y_balanced


# ============================================================
# STEP 7 — SAVE DATA
# ============================================================

def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Save train and test data separately.
    """

    logger.info("=" * 70)
    logger.info("STEP 7: SAVING PROCESSED DATA")
    logger.info("=" * 70)

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    train_data = X_train.copy()
    train_data["Label"] = y_train.values

    train_file = (
        PROCESSED_DATA_DIR /
        "train_data.csv"
    )

    train_data.to_csv(
        train_file,
        index=False
    )

    # --------------------------------------------------------
    # Testing data
    # --------------------------------------------------------

    test_data = X_test.copy()
    test_data["Label"] = y_test.values

    test_file = (
        PROCESSED_DATA_DIR /
        "test_data.csv"
    )

    test_data.to_csv(
        test_file,
        index=False
    )

    logger.info(
        f"Training data saved to:"
    )

    logger.info(
        f"  {train_file}"
    )

    logger.info(
        f"Testing data saved to:"
    )

    logger.info(
        f"  {test_file}"
    )

    logger.info(
        f"Training rows: "
        f"{len(train_data):,}"
    )

    logger.info(
        f"Testing rows: "
        f"{len(test_data):,}"
    )

    logger.info(
        f"Features: "
        f"{len(X_train.columns)}"
    )

    logger.info("=" * 70)

    return train_file, test_file


# ============================================================
# SAVE SCALER
# ============================================================

def save_scaler(scaler):
    """
    Save the fitted scaler for later real-time detection.
    """

    import joblib

    scaler_file = (
        PROCESSED_DATA_DIR /
        "scaler.pkl"
    )

    joblib.dump(
        scaler,
        scaler_file
    )

    logger.info(
        f"Scaler saved to: {scaler_file}"
    )

    return scaler_file


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def main():

    logger.info("")
    logger.info("=" * 70)
    logger.info("NIDS-ML CORRECTED PREPROCESSING PIPELINE")
    logger.info("=" * 70)
    logger.info("")

    try:

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        df = load_dataset()

        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------

        df_clean, label_col = clean_dataset(
            df
        )

        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------

        X, y = prepare_features_and_labels(
            df_clean,
            label_col
        )

        # ----------------------------------------------------
        # STEP 4
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = split_dataset(
            X,
            y
        )

        # ----------------------------------------------------
        # STEP 5
        # ----------------------------------------------------

        (
            X_train_scaled,
            X_test_scaled,
            scaler
        ) = scale_features(
            X_train,
            X_test
        )

        # ----------------------------------------------------
        # STEP 6
        # ----------------------------------------------------

        (
            X_train_balanced,
            y_train_balanced
        ) = apply_smote_to_training_data(
            X_train_scaled,
            y_train
        )

        # ----------------------------------------------------
        # STEP 7
        # ----------------------------------------------------

        train_file, test_file = save_processed_data(
            X_train_balanced,
            X_test_scaled,
            y_train_balanced,
            y_test
        )

        # ----------------------------------------------------
        # Save scaler
        # ----------------------------------------------------

        scaler_file = save_scaler(
            scaler
        )

        # ----------------------------------------------------
        # FINAL SUMMARY
        # ----------------------------------------------------

        logger.info("")
        logger.info("=" * 70)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)

        logger.info(
            f"Training samples: "
            f"{len(X_train_balanced):,}"
        )

        logger.info(
            f"Testing samples: "
            f"{len(X_test_scaled):,}"
        )

        logger.info(
            f"Number of features: "
            f"{X_train_balanced.shape[1]}"
        )

        logger.info(
            f"Training file: "
            f"{train_file}"
        )

        logger.info(
            f"Testing file: "
            f"{test_file}"
        )

        logger.info(
            f"Scaler file: "
            f"{scaler_file}"
        )

        logger.info("")
        logger.info(
            "IMPORTANT: Test data was not used "
            "to fit the scaler or SMOTE."
        )

        logger.info("=" * 70)

        return (
            X_train_balanced,
            X_test_scaled,
            y_train_balanced,
            y_test,
            scaler
        )

    except Exception as e:

        logger.error(
            "PREPROCESSING FAILED",
            exc_info=True
        )

        raise


# ============================================================
# OBJECT-ORIENTED WRAPPER
# ============================================================

class DataPreprocessor:

    def __init__(
        self,
        data_dir=RAW_DATA_DIR
    ):

        self.data_dir = Path(data_dir)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.scaler = None

    def preprocess_pipeline(
        self,
        apply_balancing=False,
        random_state=42
    ):

        global APPLY_SMOTE
        APPLY_SMOTE = apply_balancing

        global RANDOM_STATE
        RANDOM_STATE = random_state

        df = load_dataset(
            self.data_dir
        )

        df_clean, label_col = clean_dataset(
            df
        )

        X, y = prepare_features_and_labels(
            df_clean,
            label_col
        )

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = split_dataset(
            X,
            y
        )

        (
            X_train,
            X_test,
            self.scaler
        ) = scale_features(
            X_train,
            X_test
        )

        (
            X_train,
            y_train
        ) = apply_smote_to_training_data(
            X_train,
            y_train
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.scaler
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()