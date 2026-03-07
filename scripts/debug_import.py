import signal, traceback, sys

def handler(signum, frame):
    print('TIMEOUT - dumping all threads:', flush=True)
    for tid, tframe in sys._current_frames().items():
        print(f'Thread {tid}:', flush=True)
        traceback.print_stack(tframe)
        print(flush=True)
    sys.exit(1)

signal.signal(signal.SIGALRM, handler)
signal.alarm(10)

print('Step A: torch...', flush=True)
import torch
print('Step B: transformers...', flush=True)
import transformers
print('Step C: sentence_transformers submodules...', flush=True)

# Try importing submodules one by one
import importlib
for mod_name in ['sentence_transformers.util',
                 'sentence_transformers.models',
                 'sentence_transformers.similarity_functions',
                 'sentence_transformers.SentenceTransformer']:
    print(f'  importing {mod_name}...', flush=True)
    importlib.import_module(mod_name)
    print(f'  OK', flush=True)

print('ALL DONE', flush=True)
