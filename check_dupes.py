"""
check_dupes.py — one-off diagnostic, not part of the pipeline.

Question: auth_working.txt contains ~50,280 distinct lines that appear more
than once, because preprocessing.sh Step 2 samples every 1000th row of the
FULL auth.txt (including rows Step 1 already extracted) and the two files are
concatenated without de-duplication.

2_split.py finds 82 labelled rows in the test set from 80 distinct red-team
keys. This script determines whether those 2 extra rows are duplication
artefacts or genuinely distinct authentication events (Section 4.4 of the
thesis: one key matches three rows that differ outside the match key).

Run from the project folder. Takes a couple of minutes.
"""

import pandas as pd

COLS = ['time', 'src_user', 'dst_user', 'src_computer',
        'dst_computer', 'auth_type', 'logon_type',
        'auth_orient', 'success']

print("Loading test.pkl ...")
test = pd.read_pickle('test.pkl')
attacks = test[test['label'] == 1]
print(f"Labelled attack rows in test set: {len(attacks)}")

# 1. Are any labelled attack rows identical to each other across ALL columns?
exact = attacks.duplicated(subset=COLS, keep=False)
print(f"\nLabelled attack rows that are exact duplicates of another: {exact.sum()}")

if exact.sum():
    print("\nThe duplicated attack rows:")
    print(attacks[exact].sort_values(COLS).to_string(index=False))
    n_unique = attacks.drop_duplicates(subset=COLS).shape[0]
    print(f"\nDistinct attack EVENTS: {n_unique}")
    print(f"Labelled attack ROWS  : {len(attacks)}")
    print(f"Excess rows from duplication: {len(attacks) - n_unique}")
else:
    print("\nNone. All 82 labelled rows are distinct authentication events.")
    print("The 80-keys-to-82-rows difference is therefore genuine: the extra")
    print("rows are distinct auth events that share a red-team key and differ")
    print("in a field outside the (time, src_computer, dst_computer) match key.")

# 2. Same question for the training partition
print("\nLoading train.pkl ...")
train = pd.read_pickle('train.pkl')
rt = pd.read_csv('redteam.txt', header=None,
                 names=['time', 'user', 'src_computer', 'dst_computer'])
K = ['time', 'src_computer', 'dst_computer']
keys = rt[rt['time'] <= 1252799][K].drop_duplicates()
keys['lab'] = 1
tr = train.merge(keys, on=K, how='left')
tr_att = tr[tr['lab'] == 1]
tr_exact = tr_att.duplicated(subset=COLS, keep=False)
print(f"Labelled attack rows in training set: {len(tr_att)}")
print(f"  of which exact duplicates of another: {tr_exact.sum()}")
print(f"  distinct attack events: {tr_att.drop_duplicates(subset=COLS).shape[0]}")

print("\nDone.")