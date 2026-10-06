"""
fix_scaler.py  -  rebuild the CORRECT 30-feature StandardScaler.

PROBLEM
  retrain_binary_model.py re-fitted a scaler on data that was ALREADY scaled and
  overwrote data/processed/scaler.pkl with an identity scaler (mean~0, scale=1).
  The model was trained on z-scores, so inference needs the ORIGINAL raw->z-score scaler.

FIX
  Replay the exact preprocessing (same cleaning, same split, random_state=42), fit a
  StandardScaler on the RAW training rows (30 selected features), then VERIFY that
  it reproduces the z-scores already stored in selected_test_data.csv.

USAGE (put this file in src/, next to retrain_binary_model.py)
  python fix_scaler.py
"""
import os, sys, glob, shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.environ.get("NIDS_RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))
PROCESSED_DIR = os.environ.get("NIDS_PROCESSED_DIR", os.path.join(BASE_DIR, "data", "processed"))
MODEL_DIR = os.environ.get("NIDS_MODEL_DIR", os.path.join(BASE_DIR, "models"))
FEATURE_FILE = os.path.join(MODEL_DIR, "feature_names.txt")
SCALER_FILE = os.path.join(PROCESSED_DIR, "scaler.pkl")
SELECTED_TEST = os.environ.get("NIDS_SELECTED_TEST", os.path.join(PROCESSED_DIR, "selected_test_data.csv"))
RANDOM_STATE, TEST_SIZE = 42, 0.20


def load_clean(features):
    """Chunked load; duplicate removal via row hashes (same result as df.drop_duplicates())."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    assert files, f"no CSVs in {RAW_DIR}"
    print(f"Loading {len(files)} CSV files (chunked)...")
    parts, hashes = [], []
    for f in files:
        for chunk in pd.read_csv(f, low_memory=False, chunksize=200_000):
            chunk.columns = chunk.columns.str.strip().str.replace(" ", "_", regex=False)
            hashes.append(pd.util.hash_pandas_object(chunk, index=False).values)
            small = chunk[features].apply(pd.to_numeric, errors="coerce")
            small["_y"] = (chunk["Label"].astype(str).str.strip().str.upper() != "BENIGN").astype(np.int8)
            parts.append(small)
    df = pd.concat(parts, ignore_index=True)
    keep = ~pd.Series(np.concatenate(hashes)).duplicated().values
    df = df[keep].reset_index(drop=True)
    print(f"Rows after de-duplication: {len(df):,}   (your log: 2,522,362)")
    return df


def main():
    features = [l.strip() for l in open(FEATURE_FILE, encoding="utf-8") if l.strip()]
    assert len(features) == 30, f"expected 30 features, got {len(features)}"

    df = load_clean(features)
    y = df.pop("_y")
    X = df                                   # no copies: keep memory low
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    # split on row indices only (identical to splitting X, y with the same seed)
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y.values)
    print(f"Training rows: {len(train_idx):,}   (your log: 2,017,889)")

    values = X.values
    scaler = StandardScaler()
    for start in range(0, len(train_idx), 200_000):          # chunked fit, float64 stats
        scaler.partial_fit(values[train_idx[start:start + 200_000]].astype(np.float64))
    X_test_head = values[test_idx[:2000]]

    # ---- VERIFY against z-scores stored on disk ---------------------------
    ref = pd.read_csv(SELECTED_TEST, nrows=2000)[features]
    mine = scaler.transform(X_test_head.astype(np.float64))
    max_diff = float(np.abs(ref.values - mine).max())
    print(f"\nMax |difference| vs selected_test_data.csv (first 2000 rows): {max_diff:.2e}")
    if max_diff > 1e-3:
        print("VERIFICATION FAILED - scaler NOT saved.")
        sys.exit(1)
    print("VERIFICATION PASSED - this scaler reproduces your stored z-scores.")

    out = os.environ.get("NIDS_OUT_SCALER", SCALER_FILE)
    if os.path.exists(out):
        backup = os.path.join(os.path.dirname(out), "scaler_identity_BROKEN.pkl")
        shutil.copy2(out, backup)
        print(f"Backed up old scaler -> {backup}")
    joblib.dump(scaler, out)
    print(f"Saved correct scaler -> {out}")
    print("Example means:", np.round(scaler.mean_[:3], 2), "scales:", np.round(scaler.scale_[:3], 2))


if __name__ == "__main__":
    main()