import json

with open("notebooks/01_eda_cmapss_loghub.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "val_engines = np.random.choice" in source:
            new_source = """# Split training data into train/validation by engine (prevent data leakage)
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(train_df, groups=train_df["engine_id"]))

train_split = train_df.iloc[train_idx].copy()
val_split = train_df.iloc[val_idx].copy()

print("✅ Data split by engine using GroupShuffleSplit\\n")

print("Train/Val/Test Split:")
print(f"  Train set: {len(train_split)} rows ({train_split['engine_id'].nunique()} engines)")
print(f"  Val set:   {len(val_split)} rows ({val_split['engine_id'].nunique()} engines)")
print(f"  Test set:  {len(test_df)} rows ({test_df['engine_id'].nunique()} engines)")

print(f"\\nTotal data: {len(train_split) + len(val_split) + len(test_df)} rows")

# Show split composition
total_rows = len(train_split) + len(val_split) + len(test_df)
print(f"\\nSplit percentages:")
print(f"  Train: {100 * len(train_split) / total_rows:.1f}%")
print(f"  Val:   {100 * len(val_split) / total_rows:.1f}%")
print(f"  Test:  {100 * len(test_df) / total_rows:.1f}%")"""
            cell["source"] = [line + "\n" for line in new_source.split("\n")]
            cell["source"][-1] = cell["source"][-1].strip() # remove last newline

with open("notebooks/01_eda_cmapss_loghub.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
