# ==================== PHASE 4: Feature Engineering (Pipeline) ====================

def process_data_pipeline(full_train, full_test, constant_cols=None):
    """
    Phase 4: Transforms raw sensor data into ML-ready sequences.
    1. Feature Selection: Drops constant/bad columns.
    2. Train/Val Split: 80/20 split by ENGINE ID (not random rows).
    3. Scaling: Fit StandardScaler on Train, apply to Val/Test.
    4. Windowing: Creates (samples, 30, features) sequences for LSTM.
    5. Saving: Dumps processed arrays to PROCESSED_DIR.
    """
    logger.info("🚀 Starting Phase 4: Feature Engineering Pipeline")
    
    # --- 1. Feature Selection ---
    # Define excluded columns (Metadata)
    meta_cols = [ID_COL, TIME_COL, 'RUL', TARGET, 'label_fail', 'max_cycle', 'RUL_end', 'dataset_id']
    
    # Identify available sensor columns
    available_cols = [c for c in full_train.columns if c not in meta_cols]
    
    # Filter out constant columns if provided
    if constant_cols:
        available_cols = [c for c in available_cols if c not in constant_cols]
        print(f"ℹ️  Dropped {len(constant_cols)} constant features.")
        
    print(f"✅ Final Feature Set ({len(available_cols)}): {available_cols}")
    
    # --- 2. Train / Validation Split ---
    # Split by Engine ID to prevent data leakage!
    unique_engines = full_train[ID_COL].unique()
    train_ids, val_ids = train_test_split(unique_engines, test_size=0.2, random_state=RANDOM_SEED)
    
    train_df = full_train[full_train[ID_COL].isin(train_ids)].copy()
    val_df = full_train[full_train[ID_COL].isin(val_ids)].copy()
    
    print(f"✅ Split: Train Engines={len(train_ids)}, Val Engines={len(val_ids)}")
    
    # --- 3. Scaling ---
    scaler = StandardScaler()
    
    # Fit on Train ONLY
    train_df[available_cols] = scaler.fit_transform(train_df[available_cols])
    val_df[available_cols] = scaler.transform(val_df[available_cols])
    full_test[available_cols] = scaler.transform(full_test[available_cols])
    
    print("✅ Scaling Completed (Fit on Train, applied to Val/Test).")
    
    # Save Scaler for Inference
    import joblib
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    
    # --- 4. Sliding Window (Vectorization) ---
    def create_sequences(df, features, seq_len=30):
        # Generates (Samples, Window, Feats) array
        X, y = [], []
        
        for engine_id, engine_data in df.groupby(ID_COL):
            engine_array = engine_data[features].values
            target_array = engine_data[TARGET].values
            
            # Skip short engines
            if len(engine_data) < seq_len:
                continue
                
            # Sliding Window
            # Use stride of 1
            # num_samples = len - seq_len + 1
             # Optimizing with stride_tricks or simple loop
            for i in range(len(engine_data) - seq_len + 1):
                X.append(engine_array[i : i + seq_len])
                y.append(target_array[i + seq_len - 1]) # Label is RUL at LAST step of window
                
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    print(f"⏳ Generating Sequences (Window={SEQ_LEN})...")
    X_train, y_train = create_sequences(train_df, available_cols, SEQ_LEN)
    X_val, y_val = create_sequences(val_df, available_cols, SEQ_LEN)
    
    # For Test, we usually take the LAST sequence for each engine (for Kaggle-style eval)
    # OR we can generate all sequences.
    # Let's generate ALL sequences for broad evaluation, but also keep 'last' for final metric.
    X_test, y_test = create_sequences(full_test, available_cols, SEQ_LEN)
    
    # Helper: Get Last Sequence per Engine for Test
    X_test_last = []
    y_test_last = []
    
    for engine_id, engine_data in full_test.groupby(ID_COL):
        if len(engine_data) >= SEQ_LEN:
            # Last window
            last_seq = engine_data[available_cols].values[-SEQ_LEN:]
            last_label = engine_data[TARGET].values[-1]
            X_test_last.append(last_seq)
            y_test_last.append(last_label)
            
    X_test_last = np.array(X_test_last, dtype=np.float32)
    y_test_last = np.array(y_test_last, dtype=np.float32)

    print(f"✅ Sequences Created:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape} (All windows)")
    print(f"   Test Last: {X_test_last.shape} (One per engine)")
    
    # --- 5. Save Processed Tensors ---
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    
    logger.info("💾 Saved processed arrays to data/processed/")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_test_last, y_test_last, available_cols

# EXECUTE PHASE 4
try:
    if 'full_train_df' not in locals():
         full_train_df = pd.read_parquet(PROJECT_ROOT / "data" / "interim" / "train_all_labeled.parquet")
         full_test_df = pd.read_parquet(PROJECT_ROOT / "data" / "interim" / "test_all_labeled.parquet")
    
    # Use constant_features from Phase 3 if available
    cons_feats = constant_features if 'constant_features' in locals() else None
    
    # Run Pipeline
    X_train, y_train, X_val, y_val, X_test, y_test, X_test_last, y_test_last, features = \
        process_data_pipeline(full_train_df, full_test_df, constant_cols=cons_feats)
        
    print(f"Features Used: {len(features)}")
    
except Exception as e:
    print(f"\n❌ PHASE 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
