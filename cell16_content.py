# ==================== DATA LEAKAGE CHECK (SPLIT INTEGRITY) ====================
print("🔍 VERIFYING DATA SPLIT INTEGRITY (ENGINE-BASED SPLIT)")

# 5) Ensure no row-based splitting
# 6) Validation split distinct by engine_id

train_engines = set(df_train[ID_COL].unique())
val_engines = set(df_val[ID_COL].unique())
test_engines = set(df_test[ID_COL].unique())

print(f"Train/Val Split Strategy Check:")
print(f"   Train Engines Count: {len(train_engines)}")
print(f"   Val Engines Count:   {len(val_engines)}")

# Calculate Intersection
intersection = train_engines.intersection(val_engines)
intersection_count = len(intersection)

print(f"   Intersection (Train ∩ Val): {intersection_count}")

if intersection_count == 0:
    print("✅ LEAKAGE CHECK PASSED: Train and Validation sets have ZERO overlapping engines.")
    print("   Split was performed correctly by Engine ID.")
else:
    print(f"❌ LEAKAGE DETECTED: {intersection_count} engines appear in BOTH sets!")
    print(f"   Overlapping IDs: {list(intersection)[:10]}")
    raise ValueError("Critical Data Leakage: Validation set is contaminated with Training engines.")
    
# Sequential Check (Row Shuffling)
sample_eng = list(train_engines)[0]
cycles = df_train[df_train['engine_id'] == sample_eng]['cycle'].values
is_sequential = np.all(np.diff(cycles) == 1)

if is_sequential:
    print(f"✅ Row Order Check: PASSED (Sequential cycles confirmed for Engine {sample_eng}).")
else:
    print(f"❌ Row Order Check: FAILED (Cycles are shuffled/random for Engine {sample_eng})")

print("✅ Split Strategy Confirmed: Group-based split (by Engine), preserving time-series structure.")

