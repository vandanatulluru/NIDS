"""
port_ablation.py - does the model need Destination_Port, and does it HURT generalization?

Trains the same Random Forest twice (30 features WITH the port, 29 features WITHOUT it)
under two evaluation protocols:

  1. RANDOM SPLIT (your current protocol)  -> reproduces the optimistic ~0.999 numbers
  2. LEAVE-ONE-CAPTURE-OUT                 -> train on 7 capture files, test on the 8th.
     Attack families that live in only one file (e.g. Wednesday's DoS attacks,
     Friday's PortScan) are completely unseen at training time.

Also reports a PORT-ONLY baseline (one feature!) to show how much of the score the port
alone explains.  Random Forest is scale-invariant, so no scaler is needed here.

Put in src/ and run:   python port_ablation.py
Optional environment variables (defaults = full-size run):
    NIDS_N_TRAIN   max training rows per fit        (default 300000)
    NIDS_N_TREES   trees per forest                  (default 100)
Outputs (results/): port_ablation_results.csv, port_ablation_families.csv,
                    PORT_ABLATION_REPORT.txt
"""
import os, sys, glob, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.environ.get("NIDS_RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))
MODEL_DIR = os.environ.get("NIDS_MODEL_DIR", os.path.join(BASE_DIR, "models"))
RESULT_DIR = os.environ.get("NIDS_RESULT_DIR", os.path.join(BASE_DIR, "results"))
N_TRAIN = int(os.environ.get("NIDS_N_TRAIN", 300_000))
N_TREES = int(os.environ.get("NIDS_N_TREES", 100))
SEED, PORT, THRESH = 42, "Destination_Port", 0.5
os.makedirs(RESULT_DIR, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ data
def load(features):
    """Chunked load + dedupe (same cleaning as data_preprocessing.py)."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    assert files, f"no CSVs in {RAW_DIR}"
    names = [os.path.basename(f).replace("_pcap_ISCX.csv", "").replace(".pcap_ISCX.csv", "")
             .replace("-WorkingHours", "").replace("-workingHours", "") for f in files]
    parts, hashes, labels = [], [], []
    for gi, f in enumerate(files):
        for chunk in pd.read_csv(f, low_memory=False, chunksize=200_000):
            chunk.columns = chunk.columns.str.strip().str.replace(" ", "_", regex=False)
            hashes.append(pd.util.hash_pandas_object(chunk, index=False).values)
            small = chunk[features].apply(pd.to_numeric, errors="coerce").astype(np.float32)
            lab = chunk["Label"].astype(str).str.strip().str.replace("\ufffd", "-", regex=False)
            small["_g"] = np.int8(gi)
            parts.append(small)
            labels.append(lab.values)
    df = pd.concat(parts, ignore_index=True)
    lab = np.concatenate(labels)
    keep = ~pd.Series(np.concatenate(hashes)).duplicated().values
    df, lab = df[keep].reset_index(drop=True), lab[keep]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[features] = df[features].fillna(df[features].median())
    y = (np.char.upper(lab.astype(str)) != "BENIGN").astype(np.int8)
    log(f"loaded {len(df):,} rows after de-duplication (attacks {y.mean():.1%}) from {len(files)} files")
    return df, y, pd.Series(lab).astype("category"), names


# ------------------------------------------------------------------ model
def fit_predict(X, y, tr, te, cols, rng):
    if len(tr) > N_TRAIN:
        tr = rng.choice(tr, N_TRAIN, replace=False)
    rf = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, class_weight="balanced",
                                random_state=SEED)
    rf.fit(X[tr][:, cols], y[tr])
    return rf.predict_proba(X[te][:, cols])[:, 1]


def metrics(yt, p):
    pred = p >= THRESH
    pos, neg = yt == 1, yt == 0
    out = {
        "rows": len(yt), "attacks": int(pos.sum()),
        "attack_recall": float((pred & pos).sum() / pos.sum()) if pos.any() else np.nan,
        "false_positive_rate": float((pred & neg).sum() / neg.sum()) if neg.any() else np.nan,
        "accuracy": float((pred == pos).mean()),
        "f1": float(f1_score(yt, pred, zero_division=0)) if pos.any() else np.nan,
        "roc_auc": float(roc_auc_score(yt, p)) if pos.any() and neg.any() else np.nan,
    }
    return out


def main():
    features = [l.strip() for l in open(os.path.join(MODEL_DIR, "feature_names.txt"), encoding="utf-8") if l.strip()]
    assert PORT in features and len(features) == 30
    df, y, lab, names = load(features)
    X = df[features].values
    g = df["_g"].values
    cols_with = list(range(len(features)))
    cols_without = [i for i, f in enumerate(features) if f != PORT]
    sets = {"with_port(30)": cols_with, "without_port(29)": cols_without}
    rng = np.random.RandomState(SEED)
    rows, fam_rows = [], []
    idx = np.arange(len(df))

    # ---- 1. random split ------------------------------------------------
    log("== protocol 1: random 80/20 split (your current evaluation)")
    tr, te = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=y)
    for name, cols in sets.items():
        t0 = time.time()
        m = metrics(y[te], fit_predict(X, y, tr, te, cols, rng))
        rows.append({"protocol": "random_split", "heldout": "20% random rows", "features": name, **m})
        log(f"   {name:18s} recall={m['attack_recall']:.4f} fpr={m['false_positive_rate']:.4f} auc={m['roc_auc']:.4f} ({time.time()-t0:.0f}s)")

    # port-only baseline on the same split
    dt = DecisionTreeClassifier(min_samples_leaf=20, random_state=SEED).fit(X[tr][:, [features.index(PORT)]], y[tr])
    m = metrics(y[te], dt.predict_proba(X[te][:, [features.index(PORT)]])[:, 1])
    rows.append({"protocol": "random_split", "heldout": "20% random rows", "features": "PORT ONLY (1 feature)", **m})
    log(f"   PORT ONLY          recall={m['attack_recall']:.4f} fpr={m['false_positive_rate']:.4f} auc={m['roc_auc']:.4f}")

    # ---- 2. leave-one-capture-out ---------------------------------------
    log("== protocol 2: leave-one-capture-out (held-out file never seen in training)")
    for gi, nm in enumerate(names):
        te = idx[g == gi]; trn = idx[g != gi]
        for name, cols in sets.items():
            t0 = time.time()
            p = fit_predict(X, y, trn, te, cols, rng)
            m = metrics(y[te], p)
            rows.append({"protocol": "leave_one_capture_out", "heldout": nm, "features": name, **m})
            # per-family recall inside the held-out file
            for fam in sorted(set(lab.iloc[te]) - {"BENIGN"}):
                sel = (lab.iloc[te].values == fam)
                fam_rows.append({"heldout": nm, "family": fam, "features": name, "rows": int(sel.sum()),
                                 "recall": float((p[sel] >= THRESH).mean())})
            r = "n/a (benign only)" if np.isnan(m["attack_recall"]) else f"{m['attack_recall']:.4f}"
            log(f"   {nm:36s} {name:18s} recall={r:>8s} fpr={m['false_positive_rate']:.4f} ({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULT_DIR, "port_ablation_results.csv"), index=False)
    fam = pd.DataFrame(fam_rows)
    fam.to_csv(os.path.join(RESULT_DIR, "port_ablation_families.csv"), index=False)

    # ---- report ----------------------------------------------------------
    L = ["=" * 78, "DESTINATION_PORT ABLATION", "=" * 78,
         f"train rows per fit <= {N_TRAIN:,} | trees = {N_TREES} | threshold = {THRESH}", ""]
    rs = res[res.protocol == "random_split"][["features", "attack_recall", "false_positive_rate", "accuracy", "roc_auc"]]
    L += ["1) RANDOM SPLIT", rs.to_string(index=False, float_format=lambda v: f"{v:.4f}"), ""]
    lo = res[res.protocol == "leave_one_capture_out"]
    att = lo[lo.attacks > 0]
    L += ["2) LEAVE-ONE-CAPTURE-OUT, per held-out capture (attack recall / false-positive rate / AUC)"]
    pv = att.pivot(index="heldout", columns="features", values=["attack_recall", "false_positive_rate", "roc_auc"])
    L += [pv.to_string(float_format=lambda v: f"{v:.4f}"), ""]
    mean = att.groupby("features")[["attack_recall", "false_positive_rate", "roc_auc"]].mean()
    L += ["   MEAN over the attack captures", mean.to_string(float_format=lambda v: f"{v:.4f}"), ""]
    L += ["3) PER-FAMILY RECALL when the family's capture is held out"]
    L += [fam.pivot_table(index=["heldout", "family"], columns="features", values="recall").to_string(float_format=lambda v: f"{v:.3f}"), ""]
    text = "\n".join(L)
    open(os.path.join(RESULT_DIR, "PORT_ABLATION_REPORT.txt"), "w", encoding="utf-8").write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()