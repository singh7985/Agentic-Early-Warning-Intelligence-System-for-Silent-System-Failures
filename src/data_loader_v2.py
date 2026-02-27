import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import logging

# Configure logger
logger = logging.getLogger(__name__)

def load_and_process_all_datasets():
    """
    Loads FD001, FD002, FD003, FD004 from local user directory.
    Combines them, processes RUL, and returns prepared arrays.
    """
    # EXTERNAL LOCAL DATA PATHS (USER PROVIDED)
    USER_DATA_DIR = Path("/Users/xe/Desktop/CAPSTONE KRISHNA SIR /CMaps")
    
    if not USER_DATA_DIR.exists():
        logger.error(f"❌ Local user data directory not found: {USER_DATA_DIR}")
        raise FileNotFoundError(f"Directory {USER_DATA_DIR} not found.")

    logger.info(f"✅ Detected local user data directory: {USER_DATA_DIR}")
    
    datasets = ["FD001", "FD002", "FD003", "FD004"]
    col_names = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    dfs_train = []
    dfs_test = []
    dfs_rul = []
    
    # 1. Load All Datasets
    for ds_name in datasets:
        train_path = USER_DATA_DIR / f"train_{ds_name}.txt"
        test_path = USER_DATA_DIR / f"test_{ds_name}.txt"
        rul_path = USER_DATA_DIR / f"RUL_{ds_name}.txt"
        
        if train_path.exists() and test_path.exists() and rul_path.exists():
            df_tr = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
            df_te = pd.read_csv(test_path, sep=r'\s+', header=None, names=col_names)
            df_r = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL_end'])
            
            # Add Dataset Identifier
            df_tr['dataset'] = ds_name
            df_te['dataset'] = ds_name
            df_r['dataset'] = ds_name
            
            dfs_train.append(df_tr)
            dfs_test.append(df_te)
            dfs_rul.append(df_r)
            logger.info(f"  Loaded {ds_name}: Train {df_tr.shape}, Test {df_te.shape}")
            
    if not dfs_train:
        raise ValueError("No datasets successfully loaded.")

    # 2. Combine Data
    # Note: We must be careful with engine_id. engine_1 in FD001 != engine_1 in FD002.
    # We will create a unique_id but keep engine_id for compatibility with old code if needed,
    # or re-map engine_id to be unique across the full set.
    
    # Re-mapping engine_id to be globally unique
    current_max_id = 0
    for i in range(len(dfs_train)):
        # Train
        dfs_train[i]['original_engine_id'] = dfs_train[i]['engine_id']
        dfs_train[i]['engine_id'] += current_max_id
        
        # Test 
        # Test IDs restart at 1 for each dataset.
        dfs_test[i]['original_engine_id'] = dfs_test[i]['engine_id']
        dfs_test[i]['engine_id'] += current_max_id
        
        # RUL (index-based, but we can add engine_id column to track)
        dfs_rul[i]['engine_id'] = range(current_max_id + 1, current_max_id + 1 + len(dfs_rul[i]))
        
        # Update offsets for next dataset
        # We need max of (train engines, test engines) or just cumulative count?
        # Usually train has more engines, but let's be safe and take the max index used.
        max_in_batch = max(dfs_train[i]['engine_id'].max(), dfs_test[i]['engine_id'].max())
        current_max_id = max_in_batch

    df_train_full = pd.concat(dfs_train, ignore_index=True)
    df_test_full = pd.concat(dfs_test, ignore_index=True)
    df_rul_full = pd.concat(dfs_rul, ignore_index=True)
    
    logger.info(f"Combined Data: Train {df_train_full.shape}, Test {df_test_full.shape}")

    # 3. Feature Engineering: Calculate RUL for Train
    # RUL = MaxCycle - CurrentCycle
    max_cycles = df_train_full.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.rename(columns={'cycle': 'max_cycle'}, inplace=True)
    
    df_train_full = df_train_full.merge(max_cycles, on='engine_id', how='left')
    df_train_full['RUL'] = df_train_full['max_cycle'] - df_train_full['cycle']
    df_train_full['RUL_clip'] = df_train_full['RUL'].clip(upper=125)
    
    # 4. Feature Selection
    # Drop meta columns and constant features
    meta_cols = ['engine_id', 'cycle', 'dataset', 'original_engine_id', 'RUL', 'RUL_clip', 'max_cycle']
    candidate_features = [c for c in df_train_full.columns if c not in meta_cols]
    
    # Variance Threshold
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(df_train_full[candidate_features])
    features = [candidate_features[i] for i in selector.get_support(indices=True)]
    
    # Filter specific sensors known to be useless
    ignored_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    ignored_ops = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    
    refined_features = []
    for f in features:
        is_ignored = False
        for ignored in ignored_sensors + ignored_ops:
            if f == ignored or f.startswith(f"{ignored}_"):
                is_ignored = True
                break
        if not is_ignored:
            refined_features.append(f)
    features = refined_features
    
    logger.info(f"Selected {len(features)} predictive features: {features}")

    # 5. Split Train/Val
    unique_engines = df_train_full['engine_id'].unique()
    train_eng, val_eng = train_test_split(unique_engines, test_size=0.2, random_state=42)
    
    df_train_final = df_train_full[df_train_full['engine_id'].isin(train_eng)].copy()
    df_val_final = df_train_full[df_train_full['engine_id'].isin(val_eng)].copy()
    
    # 6. Process Test Set RUL
    # For test set: RUL = RUL_end + (MaxCycle - CurrentCycle)
    # We need to compute MaxCycle per engine in Test
    test_max_cycles = df_test_full.groupby('engine_id')['cycle'].max().reset_index()
    test_max_cycles.columns = ['engine_id', 'max_cycle_test']
    
    # Merge RUL_end (from df_rul_full) into test meta
    test_meta = test_max_cycles.merge(df_rul_full[['engine_id', 'RUL_end']], on='engine_id', how='left')
    
    # Merge back to test main
    df_test_final = df_test_full.merge(test_meta, on='engine_id', how='left')
    
    df_test_final['RUL'] = df_test_final['RUL_end'] + (df_test_final['max_cycle_test'] - df_test_final['cycle'])
    df_test_final['RUL_clip'] = df_test_final['RUL'].clip(upper=125)
    
    # Create arrays
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(df_train_final[features].values)
    y_train = df_train_final['RUL_clip'].values
    
    X_val = scaler.transform(df_val_final[features].values)
    y_val = df_val_final['RUL_clip'].values
    
    # For Test, we usually predict on the LAST cycle for standard metrics
    # But let's return full sequence too
    X_test = scaler.transform(df_test_final[features].values)
    y_test = df_test_final['RUL_clip'].fillna(130).values
    
    df_test_last = df_test_final.groupby('engine_id').last().reset_index()
    X_test_last = scaler.transform(df_test_last[features].values)
    y_test_last = df_test_last['RUL_end'].clip(upper=125).values

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'X_test_last': X_test_last ,'y_test_last': y_test_last,
        'features': features,
        'scaler': scaler
    }
