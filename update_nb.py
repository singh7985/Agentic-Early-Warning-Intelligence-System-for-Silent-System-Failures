import json

with open('notebooks/03_ml_model_training.ipynb', 'r') as f:
    nb = json.load(f)

# Cell 14
cell_14_source = nb['cells'][14]["source"]
new_cell_14_source = []
skip_smape = False
for line in cell_14_source:
    if "def calculate_smape" in line:
        skip_smape = True
    if skip_smape and "return 100 * np.mean(diff)" in line:
        skip_smape = False
        continue
    if not skip_smape:
        new_cell_14_source.append(line)
nb['cells'][14]["source"] = new_cell_14_source

# Cell 16
cell_16_source = nb['cells'][16]["source"]
new_cell_16_source = []
skip_smape = False
for line in cell_16_source:
    if "def smape" in line:
        skip_smape = True
    if skip_smape and "return 100/len(y_true)" in line:
        skip_smape = False
        continue
    if "actual_smape = smape(y_true, y_pred)" in line:
        new_cell_16_source.append("    nasa_score = calculate_nasa_score(y_true, y_pred)\n")
        continue
    if "print(f\"sMAPE: {actual_smape:.4f}%\")" in line:
        new_cell_16_source.append("    print(f\"NASA Score: {nasa_score:.4f}\")\n")
        continue
    if not skip_smape:
        new_cell_16_source.append(line)
nb['cells'][16]["source"] = new_cell_16_source

with open('notebooks/03_ml_model_training.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
