import json

with open('notebooks/03_ml_model_training.ipynb', 'r') as f:
    nb = json.load(f)

baseline_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "baseline_checks",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==============================================================================\n",
        "# 👀 BASELINE CHECKS (MUST BEAT THESE)\n",
        "# ==============================================================================\n",
        "# If your model can't beat these baselines -> pipeline/split/label issue.\n",
        "\n",
        "print(\"👀 COMPUTING BASELINE METRICS...\")\n",
        "\n",
        "# Baseline A: Predict a constant (mean RUL of train)\n",
        "mean_rul_train = df_train['RUL'].mean()\n",
        "y_pred_baseline_a = np.full_like(y_val, mean_rul_train)\n",
        "\n",
        "print(\"\\n--- Baseline A (Mean RUL of Train) ---\")\n",
        "print(f\"Predicting constant RUL: {mean_rul_train:.2f}\")\n",
        "baseline_a_metrics = calculate_metrics(y_val, y_pred_baseline_a, \"Baseline A\")\n",
        "\n",
        "# Baseline B: Predict capped RUL mean per dataset subset\n",
        "# Since we are using RE001, we can just use the mean of the capped RUL\n",
        "capped_mean_rul_train = y_train.mean()\n",
        "y_pred_baseline_b = np.full_like(y_val, capped_mean_rul_train)\n",
        "\n",
        "print(\"\\n--- Baseline B (Capped Mean RUL of Train) ---\")\n",
        "print(f\"Predicting constant capped RUL: {capped_mean_rul_train:.2f}\")\n",
        "baseline_b_metrics = calculate_metrics(y_val, y_pred_baseline_b, \"Baseline B\")"
    ]
}

nb['cells'][32] = baseline_cell

with open('notebooks/03_ml_model_training.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Baseline cell updated successfully.")
