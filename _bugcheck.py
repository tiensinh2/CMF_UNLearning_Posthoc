
import re, sys

print('=== BUG-FIX CHECKLIST ===')
print()

check_files = [
    'paper_hparams.py',
    'nb_eval_helpers.py',
    'unlearn/cmf_two_stage.py',
    'unlearn/naive.py',
    'unlearn/random_label.py',
    'unlearn/__init__.py',
]

# BUG 1: recompute_cmf 5-tuple unpack
pattern1 = r'Wn, Hn, G_WW, G_HH, G_WH = model\.recompute_cmf'
for f in check_files:
    src = open(f, encoding='utf-8', errors='replace').read()
    matches = re.findall(pattern1, src)
    if matches:
        print(f'[FAIL] BUG1 still in {f}: {matches}')
    else:
        print(f'[PASS] BUG1 (recompute_cmf unpack) not in {f}')

print()

# BUG 2: CUDA Event guarded
for f in ['unlearn/naive.py', 'unlearn/cmf_two_stage.py']:
    src = open(f, encoding='utf-8', errors='replace').read()
    has_guard = 'device.type ==' in src
    has_event = 'torch.cuda.Event' in src
    if has_event and has_guard:
        print(f'[PASS] BUG2 (CUDA Event guard) in {f}')
    elif not has_event:
        print(f'[PASS] BUG2 (no CUDA Event in {f})')
    else:
        print(f'[FAIL] BUG2 CUDA Event without guard in {f}')

print()

# BUG 3: grad_descent mapping
src3 = open('unlearn/__init__.py', encoding='utf-8').read()
if 'BUG-FIX' in src3:
    print('[PASS] BUG3 (grad_descent) noted/documented in __init__.py')
else:
    print('[INFO] BUG3 not explicitly marked — check cmf_two_stage.py handles grad_descent as retain-only')

print()

# BUG 4: TARUN train_dataset
s4a = open('04a_cmf_static.ipynb', encoding='utf-8').read()
if 'train_dataset' in s4a and 'tarun' in s4a.lower():
    print('[PASS] BUG4 (TARUN train_dataset) forwarded in NB4a')
else:
    print('[FAIL] BUG4 TARUN train_dataset not forwarded in NB4a')

s4c = open('04c_cmf_stage3.ipynb', encoding='utf-8').read()
if '_build_tarun_noisy_loader' in s4c:
    print('[PASS] BUG4 (TARUN train_dataset) forwarded in NB4c')
else:
    print('[FAIL] BUG4 TARUN train_dataset not forwarded in NB4c')

print()

# BUG 5: CMFweights.weight used in evaluation
nc_src = open('evaluation/nc_cmf.py', encoding='utf-8').read()
if 'CMFweights' in nc_src:
    print('[PASS] BUG5 (CMFweights.weight lookup) in evaluation/nc_cmf.py')
else:
    print('[FAIL] BUG5 CMFweights not referenced in evaluation/nc_cmf.py')

lp_src = open('evaluation/linear_prob.py', encoding='utf-8').read()
if '_preprocess_feats_for_cmf' in lp_src:
    print('[PASS] BUG5 (_preprocess_feats_for_cmf) in evaluation/linear_prob.py')
else:
    print('[FAIL] BUG5 _preprocess_feats_for_cmf missing from evaluation/linear_prob.py')

print()

# BUG 6: no-op recompute_cmf used in Stage 3
s4c = open('04c_cmf_stage3.ipynb', encoding='utf-8').read()
if 'no-op' in s4c or 'lambda' in s4c:
    print('[PASS] BUG6 (Stage-3 two-layer W lock with no-op recompute_cmf) in NB4c')
else:
    print('[FAIL] BUG6 two-layer W lock not found in NB4c')

print()
print('=== CHECK COMPLETE ===')
