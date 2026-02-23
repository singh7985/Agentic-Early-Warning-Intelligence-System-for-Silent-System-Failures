import json

with open("notebooks/03_ml_model_training.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "def calculate_smape" in source:
            new_source = source.replace(
                "# 📊 STEP 14: ROBUST EVALUATION METRICS (RMSE, MAE, R², sMAPE)",
                "# 📊 STEP 14: ROBUST EVALUATION METRICS (RMSE, MAE, R², NASA Score)"
            ).replace(
                "# Robust Percentage Metric: sMAPE (Symmetric MAPE) or epsilon-MAPE",
                "# Robust Metric: NASA Scoring Function (Asymmetric penalty)"
            )
            
            old_func = """def calculate_smape(y_true, y_pred):
    \"""
    Symmetric Mean Absolute Percentage Error (sMAPE)
    Range: [0, 200%]
    Handles zeros better than MAPE.
    \"""
    denominator = (np.abs(y_true) + np.abs$y_pred)) / 2.0
    diff = np.abs$y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  # Handle 0/0 case
    return 100 * np.mean(diff)"""
            
            new_func = """def calculate_nasa_score(y_true, y_pred):
    \"""
    NASA Scoring Function (Asymmetric penalty)
    \"""
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))"""
            
            new_source = new_source.replace(old_func, new_func)
            
            new_source = new_source.replace(
                "smape = calculate_smape(y_true, y_pred)",
                "nasa_score = calculate_nasa_score(y_true, y_pred)"
            ).replace(
                "print(f\"RMSE : {rmse:.4f}\")",
                "print(f\"RMSE : {rmse:.4f} (off by ~{rmse:.0f} cycles)\")"
            ).replace(
                "print(f\"sMAPE: {smape:.2f}%\")",
                "print(f\"NASA Score: {nasa_score:.2f}\")"
            ).replace(
                "return {'rmse': rmse, 'mae': mae, 'r2': r2, 'smape': smape}",
                "return {'rmse': rmse, 'mae': mae, 'r2': r2, 'nasa_score': nasa_score}"
            )
            
            cell["source"] = [line + "\n" for line in new_source.split("\n")]
            if not source.endswith("\n"):
                cell["source"][-1] = cell["source"][-1].rstrip("\n")

with open("notebooks/03_ml_model_training.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully.")
