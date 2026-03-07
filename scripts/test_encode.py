import os, numpy as np
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
print('Loading SentenceTransformer...')
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
print('Model loaded')
texts = ['test query 1', 'test query 2', 'test query 3']
e = m.encode(texts, show_progress_bar=False, batch_size=64, normalize_embeddings=True)
print(f'Encoded: {e.shape}')
print('SUBPROCESS TEST PASSED')
