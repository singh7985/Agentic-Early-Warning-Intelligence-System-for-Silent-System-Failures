import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

REPORT_DIR = Path("reports/eda")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def find_file(name: str) -> str:
    hits = glob.glob(f"**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"Missing file: {name}. Put train/test/RUL files inside data/raw/CMAPSS/")
    hits = sorted(hits, key=lambda x: len(x))
    return hits[0]

def load_raw(dataset: str):
    train_path = find_file(f"train_{dataset}.txt")
    test_path  = find_file(f"test_{dataset}.txt")
    rul_path   = find_file(f"RUL_{dataset}.txt")

    cols = (
        ["engine_id", "cycle"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=cols)
    test_df  = pd.read_csv(test_path,  sep=r"\s+", header=None, names=cols)
    rul_last = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].values
    return train_df, test_df, rul_last

def add_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    mx = train_df.groupby("engine_id")["cycle"].max()
    out = train_df.copy()
    out["RUL"] = out.apply(lambda r: mx.loc[r["engine_id"]] - r["cycle"], axis=1)
    return out

def add_test_rul(test_df: pd.DataFrame, rul_last: np.ndarray) -> pd.DataFrame:
    out = test_df.copy()
    engine_ids = np.sort(out["engine_id"].unique())
    rul_last = np.array(rul_last).reshape(-1)

    if len(engine_ids) != len(rul_last):
        raise ValueError(f"Test engines={len(engine_ids)} but RUL file has {len(rul_last)} values")

    rul_map = dict(zip(engine_ids, rul_last))
    mx = out.groupby("engine_id")["cycle"].max()

    def compute(row):
        eid = row["engine_id"]
        return (mx.loc[eid] - row["cycle"]) + rul_map[eid]

    out["RUL"] = out.apply(compute, axis=1)
    return out

def clip_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    out = df.copy()
    out["RUL_clip"] = np.minimum(out["RUL"], cap)
    return out

def load_all():
    trains, tests = [], []
    engine_offset = 0

    for ds in DATASETS:
        tr, te, rul_last = load_raw(ds)
        tr = clip_rul(add_train_rul(tr), 125)
        te = clip_rul(add_test_rul(te, rul_last), 125)

        tr["dataset"] = ds
        te["dataset"] = ds

        tr["engine_id_original"] = tr["engine_id"]
        te["engine_id_original"] = te["engine_id"]

        tr["engine_id"] += engine_offset
        te["engine_id"] += engine_offset

        trains.append(tr)
        tests.append(te)

        engine_offset = max(tr["engine_id"].max(), te["engine_id"].max()) + 1

    return pd.concat(trains, ignore_index=True), pd.concat(tests, ignore_index=True)

def summarize(df: pd.DataFrame, name: str):
    lengths = df.groupby("engine_id")["cycle"].max()
    s = pd.DataFrame([{
        "name": name,
        "rows": len(df),
        "engines": df["engine_id"].nunique(),
        "min_cycle": int(df["cycle"].min()),
        "max_cycle": int(df["cycle"].max()),
        "avg_cycles_per_engine": float(lengths.mean()),
        "median_cycles_per_engine": float(lengths.median()),
        "min_cycles_per_engine": int(lengths.min()),
        "max_cycles_per_engine": int(lengths.max()),
    }])
    print("\n", s.to_string(index=False))
    s.to_csv(REPORT_DIR / f"{name}_summary.csv", index=False)

def variance_report(train_df: pd.DataFrame, var_threshold=1e-8):
    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    rows = []
    for ds, g in train_df.groupby("dataset"):
        v = g[sensor_cols].var()
        low = v[v < var_threshold].sort_values()
        rows.append({
            "dataset": ds,
            "low_var_sensors_count": int(low.shape[0]),
            "low_var_sensors": ",".join(low.index.tolist())
        })
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "low_variance_sensors_by_dataset.csv", index=False)
    print("\nSaved:", REPORT_DIR / "low_variance_sensors_by_dataset.csv")

def corr_with_rul(train_df: pd.DataFrame, method="spearman"):
    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    corr = train_df[sensor_cols + ["RUL_clip"]].corr(method=method)["RUL_clip"].drop("RUL_clip")
    corr = corr.sort_values(key=lambda s: s.abs(), ascending=False)
    out = corr.reset_index()
    out.columns = ["feature", f"{method}_corr_with_RUL_clip"]
    out.to_csv(REPORT_DIR / f"{method}_corr_with_rul_clip.csv", index=False)
    print("\nTop 10 correlated sensors:\n", out.head(10).to_string(index=False))
    return out

def hist_plot(df: pd.DataFrame, col: str, fname: str, title: str):
    plt.figure()
    plt.hist(df[col].values, bins=80)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / fname, dpi=150)
    plt.close()

def main():
    print("Loading all datasets (FD001–FD004)...")
    train_all, test_all = load_all()

    train_all.to_parquet(INTERIM_DIR / "train_all.parquet", index=False)
    test_all.to_parquet(INTERIM_DIR / "test_all.parquet", index=False)
    print("Saved:", INTERIM_DIR / "train_all.parquet")
    print("Saved:", INTERIM_DIR / "test_all.parquet")

    summarize(train_all, "train_all")
    summarize(test_all, "test_all")

    by_ds = train_all.groupby("dataset").agg(rows=("engine_id", "size"), engines=("engine_id", "nunique")).reset_index()
    by_ds.to_csv(REPORT_DIR / "train_by_dataset.csv", index=False)

    hist_plot(train_all, "RUL", "train_RUL_hist.png", "Train RUL distribution (all datasets)")
    hist_plot(train_all, "RUL_clip", "train_RUL_clip_hist.png", "Train RUL_clip distribution (cap=125)")
    hist_plot(test_all, "RUL", "test_RUL_hist.png", "Test RUL distribution (all datasets)")
    hist_plot(test_all, "RUL_clip", "test_RUL_clip_hist.png", "Test RUL_clip distribution (cap=125)")

    variance_report(train_all)
    corr_with_rul(train_all, method="spearman")

    print("\n✅ Phase 1 EDA complete. Check outputs in:", REPORT_DIR)

if __name__ == "__main__":
    main()
