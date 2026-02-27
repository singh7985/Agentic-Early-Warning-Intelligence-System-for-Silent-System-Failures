# DIAGNOSTIC CELL 4: Check RUL Definition in Test vs Train
# Check if RUL decreases as expected in Test Set
print("=== RUL Consistency Check ===")
test_engine_ids = df_test['engine_id'].unique()
sample_engine = test_engine_ids[0]

print(f"Checking Test Engine {sample_engine}...")
subset = df_test[df_test['engine_id'] == sample_engine]
print(f"Engine {sample_engine} has {len(subset)} cycles.")
print(f"First 5 RULs: {subset['RUL_clip'].head(5).values}")
print(f"Last 5 RULs:  {subset['RUL_clip'].tail(5).values}")

# Check verify if RUL is decreasing
is_decreasing = subset['RUL_clip'].is_monotonic_decreasing
print(f"Is RUL strictly decreasing? {is_decreasing}")

# Check Training Set Logic
print("\nChecking Training Set RUL Logic...")
train_subset = df_train[df_train['engine_id'] == df_train['engine_id'].iloc[0]]
print(f"First 5 Train RULs: {train_subset['RUL_clip'].head(5).values}")
print(f"Last 5 Train RULs:  {train_subset['RUL_clip'].tail(5).values}")

# Check Max RUL used in clipping
print(f"\nMax Train RUL: {y_train.max()}")
print(f"Max Test RUL:  {y_test.max()}")

