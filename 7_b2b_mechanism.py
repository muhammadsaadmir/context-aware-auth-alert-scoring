"""
7_b2b_mechanism.py

Tests the two hypotheses for why Baseline 2b (Metadata One-Hot) separates
the test attacks almost perfectly (AUROC 1.000, median attack rank 55 in the
stage-4 run). Reported as Experiments D and E in Section 5.6 of the thesis.

  (a) Identity overlap. Red-team identifiers (users, hosts) appear in both
      the training and the test partition, so the one-hot LR keys on
      identity rather than behaviour.
  (b) Training-only attack identifiers. Rare identifiers appear in training
      only in attack rows; their one-hot columns are pure attack indicators
      and act as oracles at test time.

Experiment A (hypothesis a) - identity overlap.
    Build the sets of users / src hosts / dst hosts that occur in TRAINING
    ATTACK rows. For the 82 test attack rows, count how many contain at
    least one such identifier, per role and combined, as percentages of 82.
    The same rates over the test NORMAL rows are printed alongside as the
    base-rate contrast: a membership signal is only an oracle if attacks
    carry it and normal traffic mostly does not.

Experiment B (hypothesis b) - oracle columns.
    Refit Baseline 2b exactly as 4_evaluate.py does (OneHotEncoder fitted on
    the training vocabulary, hour and fail appended, balanced logistic
    regression on the full training partition), confirm the test AUROC
    reproduces, then rank the one-hot columns by |coefficient| and inspect
    the top 20: training occurrences, attack occurrences among them, purity
    (100 percent attack in training), and presence in the 82 test attack
    rows. A full-vocabulary purity census counts every column that is 100
    percent attack in training and how many of the 82 test attacks touch at
    least one such pure column. The top-20 table is also written to
    b2b_mechanism_top20.csv for Table 5.7 of the thesis.

Run from the project folder AFTER stage 2 has produced train.pkl and
test.pkl. Nothing in the existing pipeline is modified and no existing file
is overwritten.

Runtime: comparable to one 4_evaluate.py run (the label apply and one
Baseline 2b refit dominate).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

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

train_attacks = train[y_train == 1]
test_attacks = test[y_test == 1]
test_normals = test[y_test == 0]

# ---------------------------------------------------------------- Experiment A
print("\n" + "=" * 68)
print(" EXPERIMENT A - identity overlap between training-attack rows")
print("               and the test partition (hypothesis a)")
print("=" * 68)

A_users = set(train_attacks['src_user'].dropna())
A_src = set(train_attacks['src_computer'].dropna())
A_dst = set(train_attacks['dst_computer'].dropna())
A_hosts = A_src | A_dst
print(f"\nIdentifiers in TRAINING ATTACK rows: "
      f"{len(A_users)} users, {len(A_src)} src hosts, "
      f"{len(A_dst)} dst hosts ({len(A_hosts)} distinct hosts)")


def overlap_rates(df):
    src_hit = df['src_computer'].isin(A_src)
    dst_hit = df['dst_computer'].isin(A_dst)
    user_hit = df['src_user'].isin(A_users)
    either_host = src_hit | dst_hit
    any_ident = either_host | user_hit
    n = len(df)
    return [
        ('src_computer in train-attack src set', src_hit.sum(), n),
        ('dst_computer in train-attack dst set', dst_hit.sum(), n),
        ('at least one host in train-attack host sets', either_host.sum(), n),
        ('src_user in train-attack user set', user_hit.sum(), n),
        ('any identifier overlaps (host or user)', any_ident.sum(), n),
    ]


att_rows = overlap_rates(test_attacks)
nor_rows = overlap_rates(test_normals)
print(f"\n{'membership criterion':<46} {'attacks':>14} {'normal (base rate)':>20}")
for (name, a_cnt, a_n), (_, n_cnt, n_n) in zip(att_rows, nor_rows):
    print(f"{name:<46} {a_cnt:>4}/{a_n} ={100*a_cnt/a_n:5.1f}% "
          f"{n_cnt:>12,} ={100*n_cnt/n_n:6.2f}%")

# ---------------------------------------------------------------- Experiment B
print("\n" + "=" * 68)
print(" EXPERIMENT B - oracle columns in Baseline 2b (hypothesis b)")
print("=" * 68)

# Baseline 2b, constructed exactly as in 4_evaluate.py
print("\nRefitting Baseline 2b (one-hot, training vocabulary only) ...")
ID_COLS = ['src_user', 'src_computer', 'dst_computer']
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(train[ID_COLS].fillna('UNK'))


def build_meta_onehot(df):
    cats = ohe.transform(df[ID_COLS].fillna('UNK'))
    dense = np.column_stack([df['hour'].to_numpy(),
                             (df['success'] == 'Fail').astype(int).to_numpy()])
    return hstack([cats, csr_matrix(dense)], format='csr')


Xtr = build_meta_onehot(train)
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(Xtr, y_train)
Xte = build_meta_onehot(test)
auroc = roc_auc_score(y_test, lr.predict_proba(Xte)[:, 1])
print(f"Reproduction check - test AUROC (should reproduce 1.000): {auroc:.3f}")
del Xte  # free the 39M-row matrix before the census builds more test matrices

onehot_names = ohe.get_feature_names_out(ID_COLS)
n_onehot = len(onehot_names)
names = list(onehot_names) + ['hour', 'fail']
coefs = lr.coef_[0]

# per-column training occurrence and attack counts (one-hot columns only)
Xtr_onehot = Xtr[:, :n_onehot]
total_per_col = np.asarray(Xtr_onehot.sum(axis=0)).ravel()
attack_per_col = np.asarray(Xtr_onehot[y_train.to_numpy() == 1].sum(axis=0)).ravel()
pure = (total_per_col > 0) & (attack_per_col == total_per_col)

# presence of each column in the 82 test attack rows
Xta_onehot = build_meta_onehot(test_attacks)[:, :n_onehot]
test_attack_presence = np.asarray((Xta_onehot > 0).sum(axis=0)).ravel()

print("\nTop 20 features by |coefficient|:")
header = (f"{'rank':>4} {'feature':<34} {'coef':>10} {'train_n':>9} "
          f"{'train_attack_n':>15} {'pct_attack':>11} {'pure':>5} {'in_test_attacks':>16}")
print(header)
order = np.argsort(-np.abs(coefs))
csv_rows = []
shown = 0
for idx in order:
    if shown >= 20:
        break
    shown += 1
    if idx < n_onehot:
        tn, an = int(total_per_col[idx]), int(attack_per_col[idx])
        pct = 100 * an / tn if tn else 0.0
        p = 'YES' if pure[idx] else 'no'
        ta = int(test_attack_presence[idx])
        print(f"{shown:>4} {names[idx]:<34} {coefs[idx]:>+10.3f} {tn:>9,} "
              f"{an:>15,} {pct:>10.1f}% {p:>5} {ta:>13}/{len(test_attacks)}")
        csv_rows.append([shown, names[idx], coefs[idx], tn, an, round(pct, 1),
                         p, ta])
    else:
        print(f"{shown:>4} {names[idx]:<34} {coefs[idx]:>+10.3f} "
              f"{'(dense column - occurrence stats not applicable)':>56}")
        csv_rows.append([shown, names[idx], coefs[idx], '', '', '', '', ''])

pd.DataFrame(csv_rows, columns=['rank', 'feature', 'coefficient', 'train_n',
                                'train_attack_n', 'pct_attack_in_training',
                                'pure_attack_column', 'present_in_test_attacks'
                                ]).to_csv('b2b_mechanism_top20.csv', index=False)
print("\nTop-20 table written to b2b_mechanism_top20.csv")

# full-vocabulary purity census
print("\nPurity census over the full one-hot vocabulary:")
n_pure = int(pure.sum())
print(f"  columns occurring in training:            {int((total_per_col > 0).sum()):,}")
print(f"  columns 100% attack in training (pure):   {n_pure:,}")
print(f"  pure columns with >= 2 training rows:     {int((pure & (total_per_col >= 2)).sum()):,}")
pos_mass = coefs[:n_onehot][coefs[:n_onehot] > 0].sum()
pure_mass = coefs[:n_onehot][pure & (coefs[:n_onehot] > 0)].sum()
print(f"  share of positive one-hot coefficient mass on pure columns: "
      f"{100 * pure_mass / pos_mass if pos_mass else 0:.1f}%")

# how many of the 82 test attacks (and normal rows) touch >= 1 pure column
pure_vec = csr_matrix(pure.astype(np.int8).reshape(-1, 1))
att_touch = np.asarray((Xta_onehot @ pure_vec).todense()).ravel() > 0
Xno_onehot = build_meta_onehot(test_normals)[:, :n_onehot]
nor_touch = np.asarray((Xno_onehot @ pure_vec).todense()).ravel() > 0
print(f"\n  test ATTACK rows touching >= 1 pure column: "
      f"{int(att_touch.sum())}/{len(test_attacks)} = {100*att_touch.mean():.1f}%")
print(f"  test NORMAL rows touching >= 1 pure column: "
      f"{int(nor_touch.sum()):,}/{len(test_normals):,} = {100*nor_touch.mean():.2f}%")

print("\nDone.")
