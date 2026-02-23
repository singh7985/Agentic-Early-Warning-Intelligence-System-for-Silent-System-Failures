import json

with open('notebooks/03_ml_model_training.ipynb', 'r') as f:
    nb = json.load(f)

code = []
for cell in nb['cells'][:33]:
    if cell['cell_type'] == 'code':
        code.append(''.join(cell['source']))

with open('run_baseline.py', 'w') as f:
    f.write('\n'.join(code))
