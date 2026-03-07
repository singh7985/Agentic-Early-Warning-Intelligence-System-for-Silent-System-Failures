#!/usr/bin/env python3
"""Quick test of NB07 cell 2 logic to find crashes."""
import sys, os, json, joblib, numpy as np, pandas as pd, pickle, subprocess, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

print("Step 1: Loading models and data...")
with open(MODELS_DIR / 'baseline_features.json') as f:
    feature_cols = json.load(f)
xgb_model = joblib.load(MODELS_DIR / 'xgb_rul_baseline.joblib')
rf_model = joblib.load(MODELS_DIR / 'rf_failure_baseline.joblib')
test_df = pd.read_csv(DATA_DIR / 'test_features.csv')
print(f"  OK: {len(feature_cols)} features, {test_df.shape}")

print("Step 2: Predictions...")
X_test = test_df[feature_cols].fillna(0).values
test_df['ml_rul'] = np.clip(xgb_model.predict(X_test), 1, None)
test_df['ml_fail_pred'] = rf_model.predict(X_test)
test_df['ml_fail_proba'] = rf_model.predict_proba(X_test)[:, 1]
engine_info = test_df.groupby('engine_id').agg(
    max_cycle=('cycle', 'max'), rul_at_last=('RUL_clip', 'last')
).reset_index()
engine_info['failure_cycle'] = engine_info['max_cycle'] + engine_info['rul_at_last']
print("  OK")

print("Step 3: Anomaly detection...")
raw_sensor_cols = sorted([c for c in test_df.columns
                          if c.startswith('sensor_') and '_avg_' not in c
                          and '_std_' not in c and '_roll_' not in c])
healthy_data = test_df.loc[test_df['RUL_clip'] > 100, raw_sensor_cols].values
from src.anomaly import IsolationForestDetector
ad = IsolationForestDetector(contamination=0.1)
ad.fit(healthy_data)
anom_raw = ad.decision_function(test_df[raw_sensor_cols].values)
test_df['anomaly_score'] = 1.0 / (1.0 + np.exp(anom_raw))
print(f"  OK: mean={test_df['anomaly_score'].mean():.3f}, flagged={(test_df['anomaly_score'] > 0.5).sum()}")

print("Step 4: KB + FAISS load...")
import faiss
KB_DIR = PROJECT_ROOT / 'data' / 'vector_db'
with open(KB_DIR / 'documents.pkl', 'rb') as f:
    kb_data = pickle.load(f)
kb_chunks = kb_data['chunks']
faiss_index = faiss.read_index(str(KB_DIR / 'vector_store' / 'faiss_index.bin'))
print(f"  OK: {len(kb_chunks)} chunks, {faiss_index.ntotal} vectors")

print("Step 5: Encoding queries via subprocess...")
engine_ids_unique = test_df['engine_id'].unique()
queries = []
for eid in engine_ids_unique:
    last_row = test_df[test_df['engine_id'] == eid].iloc[-1]
    sv = last_row[raw_sensor_cols].values
    top = ", ".join(f"{c}={v:.2f}" for c, v in zip(raw_sensor_cols[:6], sv[:6]))
    queries.append(f"Engine degradation analysis: {top}. Failure probability {last_row['ml_fail_proba']:.0%}.")

tmp_dir = tempfile.mkdtemp()
tp = os.path.join(tmp_dir, 'texts.json')
ep = os.path.join(tmp_dir, 'embeddings.npy')
with open(tp, 'w') as f:
    json.dump(queries, f)

script = (
    "import os, json, numpy as np;"
    "os.environ['TOKENIZERS_PARALLELISM']='false';"
    "os.environ['OMP_NUM_THREADS']='1';"
    "from sentence_transformers import SentenceTransformer;"
    f"m=SentenceTransformer('all-MiniLM-L6-v2');"
    f"ts=json.load(open('{tp}'));"
    f"e=m.encode(ts,show_progress_bar=False,batch_size=64,normalize_embeddings=True);"
    f"np.save('{ep}',e)"
)
r = subprocess.run(
    [sys.executable, '-c', script],
    capture_output=True, text=True, timeout=300,
    env={**os.environ, 'TOKENIZERS_PARALLELISM': 'false',
         'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
)
if r.returncode != 0:
    print(f"  FAILED: {r.stderr[:500]}")
    sys.exit(1)
query_embeddings = np.load(ep)
os.remove(tp); os.remove(ep); os.rmdir(tmp_dir)
print(f"  OK: {query_embeddings.shape}")

print("Step 6: FAISS search...")
k = 5
distances, indices = faiss_index.search(query_embeddings.astype(np.float32), k)
print(f"  OK: distances={distances.shape}")

print("Step 7: Engine retrieval extraction...")
engine_retrieval = {}
kb_available = True
for i, eid in enumerate(engine_ids_unique):
    valid = indices[i] >= 0
    if valid.any():
        sims = distances[i][valid]
        chunk_idx = indices[i][valid]
        end_ruls = []
        severities = []
        for ci in chunk_idx:
            if int(ci) < len(kb_chunks):
                meta = kb_chunks[int(ci)].get('metadata', {})
                er = meta.get('end_rul')
                if er is not None and float(er) > 0:
                    end_ruls.append(float(er))
                sev = meta.get('severity', 0)
                if isinstance(sev, (int, float)) and sev > 0:
                    severities.append(float(sev))
        engine_retrieval[eid] = {
            'mean_sim': float(np.mean(sims)),
            'n_results': int(valid.sum()),
            'retrieved_mean_rul': float(np.mean(end_ruls)) if end_ruls else None,
            'mean_severity': float(np.mean(severities)) if severities else 0.0,
        }
    else:
        engine_retrieval[eid] = {'mean_sim': 0.0, 'n_results': 0, 'retrieved_mean_rul': None, 'mean_severity': 0.0}
sims_all = [v['mean_sim'] for v in engine_retrieval.values()]
n_with_rul = sum(1 for v in engine_retrieval.values() if v['retrieved_mean_rul'] is not None)
print(f"  OK: mean_sim={np.mean(sims_all):.3f}, end_rul={n_with_rul}/{len(engine_retrieval)}")

print("Step 8: RAG + Agent predictions...")
test_df['rag_rul'] = test_df['ml_rul'].copy()
for eid in test_df['engine_id'].unique():
    mask = test_df['engine_id'] == eid
    info = engine_retrieval.get(eid, {})
    sim = info.get('mean_sim', 0)
    ret_rul = info.get('retrieved_mean_rul')
    severity = info.get('mean_severity', 0)
    if sim > 0.3:
        if ret_rul is not None:
            blend_w = min(sim, 0.9) * 0.33
            test_df.loc[mask, 'rag_rul'] = test_df.loc[mask, 'ml_rul'] * (1 - blend_w) + ret_rul * blend_w
        elif severity > 0:
            blend_w = min(sim, 0.9) * min(severity, 1.0) * 0.25
            test_df.loc[mask, 'rag_rul'] = test_df.loc[mask, 'ml_rul'] * (1 - blend_w)
test_df['rag_rul'] = np.clip(test_df['rag_rul'], 1, 125)
test_df['agents_rul'] = test_df['rag_rul'].copy()
amask = test_df['anomaly_score'] > 0.5
if amask.any():
    red = test_df.loc[amask, 'anomaly_score'] * 0.15
    test_df.loc[amask, 'agents_rul'] *= (1 - red)
test_df['agents_rul'] = np.clip(test_df['agents_rul'], 1, 125)
print(f"  OK: RAG changed={int((test_df['rag_rul'] != test_df['ml_rul']).sum())}, Agent changed={int((test_df['agents_rul'] != test_df['rag_rul']).sum())}")

print("Step 9: MAE results...")
for name, col in [("ML-Only", "ml_rul"), ("ML+RAG", "rag_rul"), ("ML+RAG+Agents", "agents_rul")]:
    mae = (test_df[col] - test_df['RUL_clip']).abs().mean()
    print(f"  {name:18s} MAE = {mae:.2f} cycles")

print("\nALL STEPS COMPLETED SUCCESSFULLY")
