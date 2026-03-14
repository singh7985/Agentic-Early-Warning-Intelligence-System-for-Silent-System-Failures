import json

with open("notebooks/07_system_evaluation.ipynb") as f:
    nb07 = json.load(f)
checks = []
for i, cell in enumerate(nb07["cells"]):
    src = "".join(cell["source"])
    if "should_correct_ag" in src:
        checks.append(f"NB07 Cell {i}: Agent correction fixed")
    if "no_ret_rul = ml" in src and "concordance" in src:
        checks.append(f"NB07 Cell {i}: Ablation no-retrieval fixed")
    if "Maintained or Improved" in src:
        checks.append(f"NB07 Cell {i}: Chart title fixed")
    if "concordant_risk" in src and "ml_optimistic" in src:
        checks.append(f"NB07 Cell {i}: RAG calibration fixed")

with open("notebooks/08_mlops_monitoring.ipynb") as f:
    nb08 = json.load(f)
for i, cell in enumerate(nb08["cells"]):
    src = "".join(cell["source"])
    if "WARN_H" in src and "concordant" in src:
        checks.append(f"NB08 Cell {i}: RAG calibration fixed")
    if "concordance-gated calibration" in src:
        checks.append(f"NB08 Cell {i}: Conclusions fixed")

for c in checks:
    print(c)
print(f"Total verified: {len(checks)}")
