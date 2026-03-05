"""Find the exact 90 features used by the saved baseline models."""
import pandas as pd
import numpy as np
import json
from sklearn.feature_selection import VarianceThreshold

# Load train features
tr = pd.read_csv('data/processed/train_features.csv')
print(f"Train shape: {tr.shape}")

# Exactly replicate NB02's feature selection
drop_cols = ['engine_id', 'cycle', 'RUL', 'RUL_clip', 'label_fail',
             'op_setting_1', 'op_setting_2', 'op_setting_3', 'subset']
candidates = [c for c in tr.columns
              if c not in drop_cols
              and tr[c].dtype in ['float64', 'float32', 'int64', 'int32']]
print(f"Candidates after dropping meta+ops: {len(candidates)}")

# VarianceThreshold
vt = VarianceThreshold(threshold=1e-4)
vt.fit(tr[candidates])
features = [f for f, keep in zip(candidates, vt.get_support()) if keep]
dropped = [f for f, keep in zip(candidates, vt.get_support()) if not keep]

print(f"After VT: {len(features)} features")
print(f"Dropped by VT ({len(dropped)}): {dropped}")

# Save feature list
with open('models/baseline_features.json', 'w') as f:
    json.dump(features, f)
print("Saved to models/baseline_features.json")

# Verify against model
import joblib
m = joblib.load('models/xgb_rul_baseline.joblib')
print(f"\nModel expects: {m.n_features_in_} features")
print(f"We found: {len(features)} features")
print(f"Match: {m.n_features_in_ == len(features)}")

# Quick test prediction
te = pd.read_csv('data/processed/test_features.csv', nrows=5)
X_test = te[features].values
pred = m.predict(X_test)
print(f"\nTest predictions (first 5): {pred}")
print(f"Actual RUL_clip (first 5): {te['RUL_clip'].values}")
