"""
6_baseline2_mechanism.py

Diagnostics for Baseline 2 (Metadata-Only): establishes what a logistic
regression fitted to integer-coded identity columns actually keys on, given
that it has one coefficient per column and cannot represent specific host
pairs. Reported as Experiments A to C in Section 5.6 of the thesis.

Experiment A — coefficient inspection.
    Fit Baseline 2 exactly as 4_evaluate.py does and print the five coefficients.
    If the identity columns carry near-zero weight and hour/fail dominate, the
    baseline is not using identity at all. If identity dominates, it is using a
    monotonic trend over the encoder's ordering.

Experiment B — permuted encoding.
    Refit Baseline 2 with the integer codes randomly permuted. The permutation
    destroys the alphabetical ordering while preserving every other property of
    the data: same categories, same frequencies, same labels. Repeated over
    several seeded permutations so the result is not a single draw. The
    interpretation of the permutation distribution, together with the
    identity-removed refit of Experiment C, is given in Section 5.6.

Run from the project folder AFTER 2_split.py has produced train.pkl and test.pkl.
Nothing in the existing pipeline is modified and no existing file is overwritten.

Runtime: roughly the same as 4_evaluate.py, times the number of permutations.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

N_PERMUTATIONS = 20
SEED = 42

print("Loading data ...")
train = pd.read_pickle('train.pkl')
test = pd.read_pickle('test.pkl')

train['hour'] = (train['time'] % 86400) // 3600
if 'hour' not in test.columns:
    test['hour'] = (test['time'] % 86400) // 3600

# training labels, built with the same key as 2_split.py and 4_evaluate.py
redteam = pd.read_csv('redteam.txt', header=None,
                      names=['time', 'user', 'src_computer', 'dst_computer'])
attack_keys = set(zip(redteam['time'], redteam['src_computer'],
                      redteam['dst_computer']))
train['label'] = train.apply(
    lambda r: 1 if (r['time'], r['src_computer'], r['dst_computer'])
    in attack_keys else 0, axis=1)

y_train = train['label']
y_test = test['label']
print(f"Train {len(train):,} rows, {y_train.sum()} attacks | "
      f"Test {len(test):,} rows, {y_test.sum()} attacks")

# encoders fitted across both partitions, exactly as in 4_evaluate.py
le_user, le_src, le_dst = LabelEncoder(), LabelEncoder(), LabelEncoder()
le_user.fit(pd.concat([train['src_user'], test['src_user']]).fillna('UNK'))
le_src.fit(pd.concat([train['src_computer'], test['src_computer']]).fillna('UNK'))
le_dst.fit(pd.concat([train['dst_computer'], test['dst_computer']]).fillna('UNK'))

COLS = ['user', 'src', 'dst', 'hour', 'fail']


def build_meta(df, maps=None):
    """maps: optional dict of code->permuted code, applied per identity column."""
    out = pd.DataFrame({
        'user': le_user.transform(df['src_user'].fillna('UNK')),
        'src': le_src.transform(df['src_computer'].fillna('UNK')),
        'dst': le_dst.transform(df['dst_computer'].fillna('UNK')),
        'hour': df['hour'],
        'fail': (df['success'] == 'Fail').astype(int),
    })
    if maps:
        for c, m in maps.items():
            out[c] = out[c].map(lambda v: m[v])
    return out


# ---------------------------------------------------------------- Experiment A
print("\n" + "=" * 68)
print(" EXPERIMENT A - what does Baseline 2 weight?")
print("=" * 68)

Xtr, Xte = build_meta(train), build_meta(test)
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(Xtr, y_train)
auroc_orig = roc_auc_score(y_test, lr.predict_proba(Xte)[:, 1])

print(f"\nAUROC (unpermuted, should reproduce 0.787): {auroc_orig:.3f}\n")
print("Coefficients:")
for name, coef in zip(COLS, lr.coef_[0]):
    print(f"  {name:<6} {coef:+.6f}")
print(f"  {'intercept':<6} {lr.intercept_[0]:+.6f}")

print("\nContribution of each column to the score spread, |coef| x std(column):")
contrib = {c: abs(w) * Xte[c].std() for c, w in zip(COLS, lr.coef_[0])}
total = sum(contrib.values()) or 1.0
for c, v in sorted(contrib.items(), key=lambda x: -x[1]):
    print(f"  {c:<6} {v:12.4f}   {100 * v / total:5.1f}% of total")

# ---------------------------------------------------------------- Experiment B
print("\n" + "=" * 68)
print(" EXPERIMENT B - does the advantage survive a permuted encoding?")
print("=" * 68)
print("\nThe identity codes are randomly relabelled. Categories, frequencies and")
print("labels are unchanged; only the arbitrary alphabetical ordering is destroyed.\n")

rng = np.random.default_rng(SEED)
results = []
for i in range(N_PERMUTATIONS):
    maps = {}
    for c, le in (('user', le_user), ('src', le_src), ('dst', le_dst)):
        n = len(le.classes_)
        perm = rng.permutation(n)
        maps[c] = {k: int(perm[k]) for k in range(n)}
    Xtr_p, Xte_p = build_meta(train, maps), build_meta(test, maps)
    lr_p = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_p.fit(Xtr_p, y_train)
    a = roc_auc_score(y_test, lr_p.predict_proba(Xte_p)[:, 1])
    results.append(a)
    print(f"  permutation {i + 1}: AUROC = {a:.3f}")

mean_perm = float(np.mean(results))
print(f"\n  unpermuted        : {auroc_orig:.3f}")
print(f"  permuted (mean)   : {mean_perm:.3f}   over {N_PERMUTATIONS} draws")
print(f"  difference        : {auroc_orig - mean_perm:+.3f}")

# ------------------------------------------------- Experiment C
print("\n" + "=" * 68)
print(" EXPERIMENT C - what remains without the identity columns?")
print("=" * 68)

BEHAV = ['hour', 'fail']
lr_b = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_b.fit(Xtr[BEHAV], y_train)
auroc_behav = roc_auc_score(y_test, lr_b.predict_proba(Xte[BEHAV])[:, 1])

print(f"\n  full model (user, src, dst, hour, fail) : {auroc_orig:.3f}")
print(f"  behavioural only (hour, fail)           : {auroc_behav:.3f}")
print(f"  permuted identity (mean)                : {mean_perm:.3f}")
print(f"\n  identity contribution above behavioural : {auroc_orig - auroc_behav:+.3f}")

print("\nDone.")