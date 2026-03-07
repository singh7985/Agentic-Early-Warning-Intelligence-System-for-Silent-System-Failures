import signal, traceback, sys

LOG = '/tmp/debug_import.log'

def log(msg):
    with open(LOG, 'a') as f:
        f.write(msg + '\n')
    print(msg, flush=True)

def handler(signum, frame):
    log('TIMEOUT - dumping all threads:')
    for tid, tframe in sys._current_frames().items():
        lines = traceback.format_stack(tframe)
        log(f'Thread {tid}:')
        for line in lines:
            log(line.rstrip())
    sys.exit(1)

# Clear log
with open(LOG, 'w') as f:
    f.write('')

signal.signal(signal.SIGALRM, handler)
signal.alarm(15)

log('Step A: torch...')
import torch
log(f'Step A OK: torch {torch.__version__}')

log('Step B: transformers...')
import transformers 
log(f'Step B OK: transformers {transformers.__version__}')

log('Step C: sentence_transformers.util...')
import sentence_transformers.util
log('Step C OK')

log('Step D: sentence_transformers.models...')
import sentence_transformers.models
log('Step D OK')

log('Step E: sentence_transformers.SentenceTransformer...')
from sentence_transformers import SentenceTransformer
log('Step E OK')

log('ALL DONE')
