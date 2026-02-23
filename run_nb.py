import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

with open('notebooks/03_ml_model_training.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

nb.cells = nb.cells[:28]  # Wait, let's run up to the baseline cell. It's at index 32.
