import json

with open("notebooks/04_anomaly_detection.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "X_train, X_val, y_train, y_val = train_test_split" in source:
            new_source = source.replace(
                "X_train, X_val, y_train, y_val = train_test_split(\n    X_train_fe, df_train_rul['RUL'], test_size=0.2, random_state=42\n)",
                "gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\n"
                "train_idx, val_idx = next(gss.split(X_train_fe, df_train_rul['RUL'], groups=df_train_rul['engine_id']))\n"
                "X_train, X_val = X_train_fe.iloc[train_idx], X_train_fe.iloc[val_idx]\n"
                "y_train, y_val = df_train_rul['RUL'].iloc[train_idx], df_train_rul['RUL'].iloc[val_idx]"
            )
            cell["source"] = [line + "\n" for line in new_source.split("\n")]
            cell["source"][-1] = cell["source"][-1].strip()

with open("notebooks/04_anomaly_detection.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
