import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to python path to find data_loader_v2
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from src.data_loader_v2 import load_and_process_all_datasets
except ImportError as e:
    logger.error(f"Failed to import data loader: {e}")
    sys.exit(1)

def main():
    logger.info("🚀 STARTING OPTIMIZED TRAINING PIPELINE (FD001-FD004 Combined)...")
    
    try:
        # 1. LOAD DATA
        logger.info("⏳ Loading and processing data from local directory...")
        data = load_and_process_all_datasets()
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test_last'] # Using last cycle for standard metrics
        y_test = data['y_test_last']
        features = data['features']
        
        logger.info(f"✅ Data Loaded.")
        logger.info(f"   Training Samples: {X_train.shape[0]}")
        logger.info(f"   Validation Samples: {X_val.shape[0]}")
        logger.info(f"   Test Units (Last Cycle): {X_test.shape[0]}")
        logger.info(f"   Input Features: {len(features)}")

        # 2. TRAIN XGBOOST MODEL
        # XGBoost is robust for tabular sensor data and handles non-linearities well.
        # It's also much faster to train/debug than LSTM given the environment constraints.
        logger.info("⚡ Training XGBoost Regressor...")
        
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=1,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric="rmse"
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        
        # 3. EVALUATE
        logger.info("🔍 Evaluating on Test Set...")
        y_pred = model.predict(X_test)
        
        # Clip negative predictions and strictly cap at max RUL (e.g. 130)
        y_pred = np.clip(y_pred, 0, 150)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("📊 FINAL MODEL PERFORMANCE (Combined FD001-FD004)")
        print("="*50)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")
        print("="*50 + "\n")
        
        # Save Predictions for inspection
        results_df = pd.DataFrame({
            'Actual_RUL': y_test,
            'Predicted_RUL': y_pred,
            'Error': y_pred - y_test
        })
        results_path = current_dir / "reports" / "final_predictions.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"📝 Predictions saved to {results_path}")
        
        # Save Model
        models_dir = current_dir / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "xgb_combined_optimized.json"
        
        model.save_model(model_path)
        logger.info(f"💾 Model artifact saved to {model_path}")

    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
