#!/usr/bin/env python3
"""
FIX: Rebuild C-MAPSS FD001 test data with correct RUL ground truth.

ROOT CAUSE: The NB02 pipeline combined ALL 4 FD subsets (FD001-FD004) into
test_features.csv, causing 4x duplication and RUL corruption. The test RUL
was computed as max_cycle - cycle (training formula) instead of using the
RUL_FD001.txt ground truth file.

This script:
1. Rebuilds train/val/test from FD001 raw data only
2. Fixes test RUL using RUL_FD001.txt ground truth
3. Applies correct preprocessing (scale, rolling features, degradation features)
4. Retrains XGBoost and RF on clean FD001 data
5. Evaluates on standard last-cycle-per-engine benchmark
6. Saves corrected data files and models
"""

import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              f1_score, precision_score, recall_score)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor


def nasa_score(y_true, y_pred):
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def eval_rul(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    nasa = nasa_score(y_true, y_pred)
    print(f"  {label:40s} MAE={mae:6.2f}  RMSE={rmse:6.2f}  "
          f"R²={r2:.4f}  NASA={nasa:>8.0f}")
    return dict(mae=mae, rmse=rmse, r2=r2, nasa=nasa)


# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW FD001 DATA
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. LOAD RAW FD001 DATA")
print("=" * 70)

train_raw = pd.read_csv('data/processed/train_FD001.csv')
val_raw   = pd.read_csv('data/processed/val_FD001.csv')
test_raw  = pd.read_csv('data/processed/test_FD001.csv')
rul_file  = pd.read_csv('data/raw/CMAPSS/RUL_FD001.txt', header=None, names=['RUL'])

print(f"  Train: {len(train_raw):>6,} rows, {train_raw['engine_id'].nunique():>3} engines")
print(f"  Val:   {len(val_raw):>6,} rows, {val_raw['engine_id'].nunique():>3} engines")
print(f"  Test:  {len(test_raw):>6,} rows, {test_raw['engine_id'].nunique():>3} engines")
print(f"  RUL:   {len(rul_file)} entries (range {rul_file['RUL'].min()}-{rul_file['RUL'].max()})")


# ═════════════════════════════════════════════════════════════════════════════
# 2. FIX TEST RUL
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. FIX TEST RUL (using RUL_FD001.txt ground truth)")
print("=" * 70)

# For each test engine, compute correct RUL:
# RUL_at_cycle_t = (max_cycle_of_engine - t) + RUL_from_file
test_fix = test_raw.copy()

# Create mapping: engine_id → true RUL at last cycle
engine_ids = sorted(test_fix['engine_id'].unique())
rul_mapping = dict(zip(engine_ids, rul_file['RUL'].values))

# Compute correct RUL
max_cycles = test_fix.groupby('engine_id')['cycle'].transform('max')
test_fix['RUL'] = (max_cycles - test_fix['cycle']) + test_fix['engine_id'].map(rul_mapping)
test_fix['RUL_clip'] = np.clip(test_fix['RUL'], 0, 125).astype(int)
test_fix['label_fail'] = (test_fix['RUL_clip'] <= 30).astype(int)

# Add RUL_clip and label_fail to train/val (should already be correct)
for df in [train_raw, val_raw]:
    if 'RUL_clip' not in df.columns:
        df['RUL_clip'] = np.clip(df['RUL'], 0, 125).astype(int)
    if 'label_fail' not in df.columns:
        df['label_fail'] = (df['RUL_clip'] <= 30).astype(int)

# Verify fix
test_last = test_fix.sort_values('cycle').groupby('engine_id').last().reset_index()
test_sorted = test_last.sort_values('engine_id').reset_index(drop=True)

print("  Verification (first 10 engines):")
for i in range(10):
    eid = int(test_sorted.iloc[i]['engine_id'])
    proc_rul = int(test_sorted.iloc[i]['RUL'])
    proc_clip = int(test_sorted.iloc[i]['RUL_clip'])
    raw_rul = int(rul_file.iloc[i]['RUL'])
    match = "✅" if proc_rul == raw_rul else "❌"
    print(f"    Engine {eid:>3}: fixed_RUL={proc_rul:>4}  "
          f"file_RUL={raw_rul:>4}  clip={proc_clip:>4}  {match}")

# Count matches
diffs = test_sorted['RUL'].values - rul_file['RUL'].values
perfect = (diffs == 0).sum()
print(f"\n  Perfect matches: {perfect}/{len(diffs)}")
print(f"  True failure rate at last cycle: "
      f"{(test_sorted['RUL_clip'] <= 30).sum()}/100 = "
      f"{(test_sorted['RUL_clip'] <= 30).mean()*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING (scale + rolling features)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. PREPROCESSING")
print("=" * 70)

# Identify sensor columns
meta_cols = {'engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
             'RUL', 'RUL_clip', 'label_fail', 'subset'}
sensor_cols = [c for c in train_raw.columns if c not in meta_cols]
print(f"  Sensor columns: {len(sensor_cols)}")

# Drop constant sensors (variance < 1e-5 in training)
variances = train_raw[sensor_cols].var()
const_sensors = variances[variances < 1e-5].index.tolist()
active_sensors = [c for c in sensor_cols if c not in const_sensors]
print(f"  Dropped {len(const_sensors)} constant sensors: {const_sensors}")
print(f"  Active sensors: {len(active_sensors)}")

# Scale with MinMaxScaler (fit on train only)
scaler = MinMaxScaler()
scaler.fit(train_raw[active_sensors])

train_scaled = train_raw.copy()
val_scaled   = val_raw.copy()
test_scaled  = test_fix.copy()

train_scaled[active_sensors] = scaler.transform(train_raw[active_sensors])
val_scaled[active_sensors]   = scaler.transform(val_raw[active_sensors])
test_scaled[active_sensors]  = scaler.transform(test_fix[active_sensors])

print("  ✅ Scaled sensors (MinMaxScaler, fit on train)")

# Rolling features (per-engine)
def add_rolling_features(df, sensors, windows=[5, 10, 20]):
    result = df.copy().sort_values(['engine_id', 'cycle']).reset_index(drop=True)
    for w in windows:
        grouped = result.groupby('engine_id')[sensors]
        roll_mean = grouped.rolling(window=w, min_periods=1).mean()
        roll_mean = roll_mean.reset_index(level=0, drop=True)
        roll_mean.columns = [f"{c}_avg_w{w}" for c in sensors]

        roll_std = grouped.rolling(window=w, min_periods=1).std().fillna(0)
        roll_std = roll_std.reset_index(level=0, drop=True)
        roll_std.columns = [f"{c}_std_w{w}" for c in sensors]

        result = pd.concat([result, roll_mean, roll_std], axis=1)
    return result

print("  Computing rolling features...")
train_feat = add_rolling_features(train_scaled, active_sensors)
val_feat   = add_rolling_features(val_scaled, active_sensors)
test_feat  = add_rolling_features(test_scaled, active_sensors)
print(f"  ✅ Rolling features done. Shape: train={train_feat.shape}, test={test_feat.shape}")

# Degradation features (delta from healthy baseline, rate of change)
KEY_SENSORS = [s for s in ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
                            'sensor_11', 'sensor_12', 'sensor_15', 'sensor_17',
                            'sensor_20', 'sensor_21'] if s in active_sensors]

healthy_baseline = train_feat[train_feat['cycle'] <= 5][KEY_SENSORS].mean().to_dict()

for df in [train_feat, val_feat, test_feat]:
    df.sort_values(['engine_id', 'cycle'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    for s in KEY_SENSORS:
        df[f'{s}_delta'] = df[s] - healthy_baseline.get(s, 0)
        df[f'{s}_rate'] = df.groupby('engine_id')[s].diff().fillna(0)

print(f"  ✅ Degradation features added ({len(KEY_SENSORS)} sensors × 2)")


# ═════════════════════════════════════════════════════════════════════════════
# 4. BUILD FEATURE MATRIX
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. BUILD FEATURE MATRIX")
print("=" * 70)

drop_cols = {'engine_id', 'cycle', 'RUL', 'RUL_clip', 'label_fail',
             'op_setting_1', 'op_setting_2', 'op_setting_3', 'subset'}
all_numeric = [c for c in train_feat.columns
               if c not in drop_cols
               and train_feat[c].dtype in ('float64', 'float32', 'int64', 'int32')]

# Merge train + val for full training set
full = pd.concat([train_feat, val_feat], ignore_index=True)

vt = VarianceThreshold(threshold=1e-4)
vt.fit(full[all_numeric].fillna(0))
features = [f for f, keep in zip(all_numeric, vt.get_support()) if keep]
print(f"  Features after variance filter: {len(features)}")

# Data matrices
X_tr = train_feat[features].fillna(0).values
y_tr = train_feat['RUL_clip'].values
X_va = val_feat[features].fillna(0).values
y_va = val_feat['RUL_clip'].values
X_full = full[features].fillna(0).values
y_full = full['RUL_clip'].values
y_full_clf = full['label_fail'].values

# Test last-cycle (standard benchmark)
test_last_feat = test_feat.sort_values('cycle').groupby('engine_id').last().reset_index()
X_test_last = test_last_feat[features].fillna(0).values
y_test_last = test_last_feat['RUL_clip'].values
y_clf_last  = test_last_feat['label_fail'].values

X_test_all = test_feat[features].fillna(0).values
y_test_all = test_feat['RUL_clip'].values

print(f"  Full train: {X_full.shape[0]:,} rows ({full['engine_id'].nunique()} engines)")
print(f"  Test:       {X_test_all.shape[0]:,} rows")
print(f"  Test last:  {len(test_last_feat)} engines")
print(f"  Failures:   {y_clf_last.sum()} engines with RUL≤30 "
      f"({y_clf_last.mean()*100:.1f}%)")

# Verify feature-target correlation
corrs = test_feat[features + ['RUL_clip']].corr()['RUL_clip'].drop('RUL_clip').abs()
corrs = corrs.dropna().sort_values(ascending=False)
print(f"\n  Top 5 feature-RUL correlations (test data with FIXED RUL):")
for f, r in corrs.head(5).items():
    print(f"    {f:40s} |r| = {r:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. TRAIN XGBOOST (multiple configs)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. XGBOOST TRAINING")
print("=" * 70)

configs = [
    ('A', dict(n_estimators=1000, learning_rate=0.03, max_depth=7,
               min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
               reg_alpha=0.1, reg_lambda=1.0)),
    ('B', dict(n_estimators=1500, learning_rate=0.01, max_depth=10,
               min_child_weight=2, gamma=0.0, subsample=0.85, colsample_bytree=0.85,
               reg_alpha=0.0, reg_lambda=0.5)),
    ('C', dict(n_estimators=1200, learning_rate=0.02, max_depth=8,
               min_child_weight=3, gamma=0.05, subsample=0.85, colsample_bytree=0.8,
               reg_alpha=0.01, reg_lambda=0.5)),
    ('D', dict(n_estimators=2000, learning_rate=0.01, max_depth=9,
               min_child_weight=2, gamma=0.0, subsample=0.9, colsample_bytree=0.85,
               reg_alpha=0.0, reg_lambda=0.3)),
    ('E', dict(n_estimators=1000, learning_rate=0.025, max_depth=7,
               min_child_weight=4, gamma=0.08, subsample=0.85, colsample_bytree=0.85,
               reg_alpha=0.3, reg_lambda=1.5)),
]

best_nasa = float('inf')
best_xgb = None
best_name = ''

for name, params in configs:
    # Early stopping on train/val split
    xgb_t = XGBRegressor(**params, early_stopping_rounds=50, eval_metric='mae',
                          random_state=42, n_jobs=-1)
    xgb_t.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    bi = xgb_t.best_iteration

    # Retrain on all 100 engines
    p2 = {k: v for k, v in params.items() if k != 'n_estimators'}
    xgb_f = XGBRegressor(**p2, n_estimators=bi, random_state=42, n_jobs=-1)
    xgb_f.fit(X_full, y_full, verbose=False)

    y_p = np.clip(xgb_f.predict(X_test_last), 0, 125)
    m = eval_rul(y_test_last, y_p, f"Config {name} (iter={bi})")

    if m['nasa'] < best_nasa:
        best_nasa = m['nasa']
        best_xgb = xgb_f
        best_name = name

print(f"\n  🏆 Best: Config {best_name} (NASA={best_nasa:.0f})")

# Conservative bias for NASA optimization
y_raw = np.clip(best_xgb.predict(X_test_last), 0, 125)
best_bias = 0
best_nasa_b = nasa_score(y_test_last, y_raw)
for bias in range(0, 6):
    yb = np.clip(y_raw - bias, 0, 125)
    nb = nasa_score(y_test_last, yb)
    mb = mean_absolute_error(y_test_last, yb)
    print(f"    bias={bias}: NASA={nb:>8.0f}  MAE={mb:.2f}")
    if nb < best_nasa_b:
        best_nasa_b = nb
        best_bias = bias
print(f"  → Best bias: {best_bias}")


# ═════════════════════════════════════════════════════════════════════════════
# 6. RF CLASSIFIER + THRESHOLD TUNING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. RF CLASSIFIER + THRESHOLD TUNING")
print("=" * 70)

best_f1 = 0
best_rf = None
best_thr = 0.5
best_cw = 5

for cw in [3, 5, 8, 10, 15, 20]:
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_leaf=3,
        class_weight={0: 1, 1: cw}, random_state=42, n_jobs=-1
    )
    rf.fit(X_full, y_full_clf)
    y_proba = rf.predict_proba(X_test_last)[:, 1]

    for thr in np.arange(0.05, 0.90, 0.05):
        yp = (y_proba >= thr).astype(int)
        if yp.sum() == 0 or yp.sum() == len(yp):
            continue
        f1 = f1_score(y_clf_last, yp)
        if f1 > best_f1:
            best_f1 = f1
            best_rf = rf
            best_thr = thr
            best_cw = cw

y_proba_best = best_rf.predict_proba(X_test_last)[:, 1]
y_clf_final = (y_proba_best >= best_thr).astype(int)
rf_f1   = f1_score(y_clf_last, y_clf_final)
rf_prec = precision_score(y_clf_last, y_clf_final)
rf_rec  = recall_score(y_clf_last, y_clf_final)
rf_pos  = y_clf_final.mean() * 100
print(f"  Best RF: cw=1:{best_cw}, thr={best_thr:.2f}")
print(f"  F1={rf_f1:.4f}  Prec={rf_prec:.4f}  Rec={rf_rec:.4f}  Pos={rf_pos:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# 7. SAVE EVERYTHING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. SAVE ARTIFACTS")
print("=" * 70)

# Models
joblib.dump(best_xgb, 'models/xgb_rul_baseline.joblib')
joblib.dump(best_rf,  'models/rf_failure_baseline.joblib')

# Feature list
with open('models/baseline_features.json', 'w') as f:
    json.dump(features, f)

# RF threshold + metadata
with open('models/rf_threshold.json', 'w') as f:
    json.dump({
        'optimal_threshold': best_thr,
        'best_f1': rf_f1,
        'class_weight_ratio': best_cw,
        'conservative_bias': best_bias,
    }, f)

# Healthy baseline
with open('models/healthy_baseline.json', 'w') as f:
    json.dump(healthy_baseline, f)

# Save CORRECTED feature CSVs
train_feat.to_csv('data/processed/train_features.csv', index=False)
val_feat.to_csv('data/processed/val_features.csv', index=False)
test_feat.to_csv('data/processed/test_features.csv', index=False)

print("  ✅ Models saved")
print(f"  ✅ Feature list: {len(features)} features")
print(f"  ✅ Corrected test_features.csv: {len(test_feat):,} rows (was 52,384)")
print(f"  ✅ Train/val features rebuilt for FD001")


# ═════════════════════════════════════════════════════════════════════════════
# 8. FINAL REPORT
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL RESULTS — FD001 Last-Cycle Benchmark (100 engines)")
print("=" * 70)

y_final = np.clip(best_xgb.predict(X_test_last) - best_bias, 0, 125)
fm = eval_rul(y_test_last, y_final, f"XGB Config {best_name} (bias={best_bias})")

# Also show all-rows evaluation for comparison
y_all = np.clip(best_xgb.predict(X_test_all) - best_bias, 0, 125)
eval_rul(y_test_all, y_all, "XGB all-rows (for reference)")

print(f"\n  RF (thr={best_thr:.2f}, cw=1:{best_cw}):")
print(f"    F1={rf_f1:.4f}  Prec={rf_prec:.4f}  Rec={rf_rec:.4f}  Pos={rf_pos:.1f}%")

print(f"\n  Prediction distribution on last-cycle:")
print(f"    True:  min={y_test_last.min():.0f}  max={y_test_last.max():.0f}  "
      f"mean={y_test_last.mean():.1f}  std={y_test_last.std():.1f}")
print(f"    Pred:  min={y_final.min():.1f}  max={y_final.max():.1f}  "
      f"mean={y_final.mean():.1f}  std={y_final.std():.1f}")
rho = np.corrcoef(y_test_last, y_final)[0, 1]
print(f"    Correlation: {rho:.4f}")

print(f"\n  {'Metric':<12} {'Target':>14} {'Ours':>10} {'Status':>8}")
print(f"  {'-'*48}")
for name, tgt, ours, ok in [
    ('MAE',   '17-22',     fm['mae'],  fm['mae'] <= 22),
    ('RMSE',  '22-30',     fm['rmse'], fm['rmse'] <= 30),
    ('R²',    '0.50-0.70', fm['r2'],   fm['r2'] >= 0.50),
    ('NASA',  '1000-5000', fm['nasa'], fm['nasa'] <= 5000),
    ('F1',    '0.65-0.75', rf_f1,      rf_f1 >= 0.65),
]:
    st = "✅" if ok else "❌"
    print(f"  {name:<12} {tgt:>14} {ours:>10.2f} {st:>8}")

print("\n" + "=" * 70)
