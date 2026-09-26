"""
9_sensitivityC.py — Design C, the density-equalised sensitivity analysis (Section 5.7)

Question: is the Context LR vs baseline comparison confounded by the
sampling regime of auth_working.txt (complete histories for accounts
matching the 91 grep-listed substrings vs 1-in-1000 for everyone else)?

Design C equalises the information density of the PER-USER FEATURE
INPUTS without changing the data being scored:

  - For feature computation ONLY, the training history of every
    DENSE-regime user (src_user containing one of the 91 substrings,
    the same classification as 8_sampling_step0.py) is thinned to a
    per-user systematic 1-in-1000 sample: rows sorted by time, keep
    sequence positions 0, 1000, 2000, ... of each user's own events.
    SPARSE-regime users' rows are kept in full (already ~1/1000 density).
  - Scoring rows are unchanged: the full fit window trains the context
    LR, the full 39,119,508-row test set is scored, all 82 test
    positives retained, labels unchanged. Baselines are not recomputed
    (they do not consume per-user history).
  - All parameters identical to the corrected pipeline (3_features.py, 4_evaluate.py):
    circmean(high=24, low=0), circular distance, T_REF = 1002240,
    reference-window lookups for the fit window, unseen-user fallbacks
    (propensity 0 / hour 12 / novelty 0), MinMaxScaler fitted on the
    fit-window features, LogisticRegression(max_iter=1000,
    class_weight='balanced'), equal weights 0.25.

Outputs:
  - Console report in the 4_evaluate.py format for the two Design C
    context scorers, plus the point ablation block.
  - test_scored_c.pkl — slim frame (label + the two Design C scores),
    original row index preserved. NO existing file is overwritten.
  - Paired bootstrap against the ORIGINAL scores from test_scored.pkl,
    same protocol and seeds as 5_bootstrap.py (rng 42, 200,000-negative
    subsample, 1,000 resamples, per-iteration seeds i and i+10000):
    AUROC CIs for Design C and original scorers and a paired CI for the
    difference (Design C − original).
  - The decision quantity: point ΔAUROC for Context LR (Design C − original)
    on the FULL test set, read against a rule fixed before this script was
    run: |Δ| < 0.02 no material confound; 0.02 to 0.05 moderate confound,
    disclosed as a bounded limitation; > 0.05 substantial confound.
"""

import gc
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import circmean

PATTERNS = [
    "U748", "U66", "U737", "U293", "U1723", "U3635", "U162", "U3005", "U8946", "U8601",
    "U218", "U342", "U4448", "U636", "U825", "U1653", "U4978", "U5087", "U9947", "U9763",
    "U4353", "U1450", "U374", "U2575", "U882", "U8777", "U3718", "U314", "U642", "U6572",
    "U2837", "U349", "U1600", "U250", "U4856", "U9407", "U4112", "U7375", "U7507", "U415",
    "U1145", "U1480", "U453", "U207", "U1289", "U1519", "U3486", "U1592", "U1025", "U9263",
    "U655", "U86", "U3549", "U8170", "U679", "U7311", "U524", "U1133", "U78", "U3764",
    "U212", "U995", "U795", "U6691", "U2231", "U7594", "U114", "U1106", "U3575", "U3206",
    "U227", "U1306", "U8840", "U1467", "U3406", "U10379", "U8168", "U3277", "U7761",
    "U7004", "U7394", "U1048", "U5254", "U6764", "U1569", "U1581", "U1789", "U13",
    "U12", "U20", "U24",
]
assert len(PATTERNS) == 91
T_REF = 1002240
THIN = 1000

print("Loading data ...")
train = pd.read_pickle('train.pkl')
test = pd.read_pickle('test.pkl')

train['hour'] = (train['time'] % 86400) // 3600
if 'hour' not in test.columns:
    test['hour'] = (test['time'] % 86400) // 3600

# training labels, same construction as 4_evaluate.py
redteam = pd.read_csv('redteam.txt', header=None,
                      names=['time', 'user', 'src_computer', 'dst_computer'])
attack_keys = set(zip(redteam['time'], redteam['src_computer'],
                      redteam['dst_computer']))
train['label'] = train.apply(
    lambda r: 1 if (r['time'], r['src_computer'], r['dst_computer'])
    in attack_keys else 0, axis=1)
y_test = test['label']
print(f"Train {len(train):,} rows, {train['label'].sum()} attacks | "
      f"Test {len(test):,} rows, {y_test.sum()} attacks")

# dense-regime classification (same rule as 8_sampling_step0.py)
regex = re.compile("|".join(PATTERNS))
uniq = train['src_user'].dropna().unique()
dense_users = {u for u in uniq if regex.search(str(u))}
print(f"Dense-regime users in training: {len(dense_users):,} "
      f"of {len(uniq):,} distinct src_user values")


def thin_view(df, name):
    """Per-user systematic 1-in-THIN thinning of dense users' rows, in time order."""
    d = df.sort_values('time', kind='mergesort')
    is_dense = d['src_user'].isin(dense_users)
    pos = d.groupby('src_user').cumcount()
    keep = (~is_dense) | (pos % THIN == 0)
    out = d[keep]
    print(f"  {name}: kept {len(out):,} of {len(df):,} rows "
          f"(dense rows {int(is_dense.sum()):,} -> {int((is_dense & keep).sum()):,})")
    return out


def lookups(view):
    total = view.groupby('src_user').size().rename('total_events')
    fail = (view[view['success'] == 'Fail']
            .groupby('src_user').size().rename('fail_events'))
    hist = (fail / total).rename('historical_fail_propensity').fillna(0)
    mh = (view.groupby('src_user')['hour']
          .agg(lambda h: circmean(h, high=24, low=0)).rename('mean_hour'))
    ks = view.groupby('src_user')['src_computer'].apply(set).rename('known_src')
    kd = view.groupby('src_user')['dst_computer'].apply(set).rename('known_dst')
    return hist, mh, ks, kd


def add_features(df, hist, mh, ks, kd):
    df = df.join(hist, on='src_user')
    df = df.join(mh, on='src_user')
    df = df.join(ks, on='src_user')
    df = df.join(kd, on='src_user')
    df['historical_fail_propensity'] = df['historical_fail_propensity'].fillna(0)
    df['mean_hour'] = df['mean_hour'].fillna(12)
    hd = (df['hour'] - df['mean_hour']).abs()
    df['time_deviation'] = np.minimum(hd, 24 - hd)
    df['src_novelty'] = df.apply(
        lambda r: 1 if isinstance(r['known_src'], set)
        and r['src_computer'] not in r['known_src'] else 0, axis=1)
    df['dst_novelty'] = df.apply(
        lambda r: 1 if isinstance(r['known_dst'], set)
        and r['dst_computer'] not in r['known_dst'] else 0, axis=1)
    return df


# ── stage-3 equivalent: TEST features from the thinned FULL-training view ──
print("\nBuilding thinned full-training feature view ...")
full_view = thin_view(train, "full-training view")
hist, mh, ks, kd = lookups(full_view)
del full_view
gc.collect()
print("Applying Design C features to test set ...")
test = add_features(test, hist, mh, ks, kd)
del hist, mh, ks, kd
gc.collect()

# ── stage-4 equivalent: fit-window features from the thinned REFERENCE view ──
print("\nBuilding thinned reference-window feature view ...")
reference = train[train['time'] < T_REF]
fit = train[train['time'] >= T_REF].copy()
print(f"Reference window: {len(reference):,} rows   Fit window: {len(fit):,} rows   "
      f"Attacks in fit window: {fit['label'].sum()}")
ref_view = thin_view(reference, "reference-window view")
hist, mh, ks, kd = lookups(ref_view)
del ref_view, reference
gc.collect()
fit = add_features(fit, hist, mh, ks, kd)
del hist, mh, ks, kd, train
gc.collect()

context_cols = ['historical_fail_propensity', 'time_deviation',
                'src_novelty', 'dst_novelty']
scaler = MinMaxScaler()
fit[context_cols] = scaler.fit_transform(fit[context_cols].fillna(0))
test[context_cols] = scaler.transform(test[context_cols].fillna(0))

w = 0.25
test['score_ctx_equal_c'] = w * (
    test['historical_fail_propensity'] + test['time_deviation'] +
    test['src_novelty'] + test['dst_novelty'])

print("\nTraining context-aware logistic regression (Design C features)...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(fit[context_cols], fit['label'])
test['score_ctx_lr_c'] = lr.predict_proba(test[context_cols])[:, 1]
print("Learned context feature weights (Design C):")
for name, coef in zip(context_cols, lr.coef_[0]):
    print(f"  {name:<18} {coef:+.3f}")
del fit
gc.collect()


# ── evaluation, same metric functions as 4_evaluate.py ─────────────────────
def precision_at_k(scores, labels, k):
    top_k_idx = scores.nlargest(k).index
    return labels.loc[top_k_idx].sum() / k


def median_attack_rank(scores, labels):
    ranks = scores.rank(ascending=False)
    return ranks[labels == 1].median(), len(scores)


def full_report(name, score_col):
    scores = test[score_col]
    auroc = roc_auc_score(y_test, scores)
    p20 = precision_at_k(scores, y_test, 20)
    p100 = precision_at_k(scores, y_test, 100)
    p1000 = precision_at_k(scores, y_test, 1000)
    med_rank, total = median_attack_rank(scores, y_test)
    print(f"\n{name}")
    print(f"  AUROC:              {auroc:.3f}")
    print(f"  Precision@20:       {p20:.3f}")
    print(f"  Precision@100:      {p100:.3f}")
    print(f"  Precision@1000:     {p1000:.3f}")
    print(f"  Median attack rank: {med_rank:,.0f} out of {total:,}")


print("\n=== DESIGN C RESULTS ===")
full_report("Context-Aware (Equal Weights) — Design C", "score_ctx_equal_c")
full_report("Context-Aware (Logistic Regression) — Design C", "score_ctx_lr_c")

print("\n=== ABLATION (equal-weight Design C score, point values) ===")
base_auroc = roc_auc_score(y_test, test['score_ctx_equal_c'])
print(f"Full model — AUROC={base_auroc:.3f}")
for drop_feat in context_cols:
    score = test['score_ctx_equal_c'] - w * test[drop_feat]
    auroc = roc_auc_score(y_test, score)
    print(f"Without {drop_feat:<28} AUROC={auroc:.3f} (Δ{auroc - base_auroc:+.3f})")

# slim save; original index preserved; nothing overwritten
slim = test[['label', 'score_ctx_equal_c', 'score_ctx_lr_c']].copy()
slim.to_pickle('test_scored_c.pkl')
print("\nSaved test_scored_c.pkl (slim; no existing file overwritten)")
del test
gc.collect()

# ── paired bootstrap vs original scores ────────────────────────────────────
print("Loading original scores from test_scored.pkl ...")
orig_full = pd.read_pickle('test_scored.pkl')
orig = orig_full[['label', 'score_ctx_equal', 'score_ctx_lr']].copy()
del orig_full
gc.collect()

assert len(orig) == len(slim), "row-count mismatch"
assert (orig.index == slim.index).all(), "index mismatch between original and Design C frames"
assert (orig['label'].to_numpy() == slim['label'].to_numpy()).all(), "label mismatch"

df = slim.join(orig[['score_ctx_equal', 'score_ctx_lr']])
del slim, orig
gc.collect()

y = df['label']
attacks = df[y == 1]
normals = df[y == 0]
NEG_SAMPLE_SIZE = 200_000
rng = np.random.default_rng(42)
neg_idx = rng.choice(normals.index, size=NEG_SAMPLE_SIZE, replace=False)
neg_sample = normals.loc[neg_idx]
print(f"Attacks: {len(attacks)}  Normals: {len(normals):,}  "
      f"(same 200,000-negative protocol, seed 42)")

point = {c: roc_auc_score(y, df[c]) for c in
         ['score_ctx_lr', 'score_ctx_lr_c', 'score_ctx_equal', 'score_ctx_equal_c']}
print(f"\nPoint AUROC (full test set):")
print(f"  Context LR     original={point['score_ctx_lr']:.4f}   Design C={point['score_ctx_lr_c']:.4f}")
print(f"  Equal weights  original={point['score_ctx_equal']:.4f}   Design C={point['score_ctx_equal_c']:.4f}")

pairs = {'Context LR': ('score_ctx_lr_c', 'score_ctx_lr'),
         'Equal weights': ('score_ctx_equal_c', 'score_ctx_equal')}
res = {k: {'c': [], 'o': [], 'd': []} for k in pairs}
N_BOOTSTRAP = 1000
print(f"\nRunning {N_BOOTSTRAP} paired bootstrap resamples...")
for i in range(N_BOOTSTRAP):
    ba = attacks.sample(n=len(attacks), replace=True, random_state=i)
    bn = neg_sample.sample(n=len(neg_sample), replace=True, random_state=i + 10000)
    bs = pd.concat([ba, bn])
    bl = bs['label']
    for k, (cc, oc) in pairs.items():
        a_c = roc_auc_score(bl, bs[cc])
        a_o = roc_auc_score(bl, bs[oc])
        res[k]['c'].append(a_c)
        res[k]['o'].append(a_o)
        res[k]['d'].append(a_c - a_o)
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{N_BOOTSTRAP} done")

print("\n=== DESIGN C vs ORIGINAL — PAIRED BOOTSTRAP 95% CIs ===\n")
for k in pairs:
    for tag, arr in (('Design C', res[k]['c']), ('original', res[k]['o']),
                     ('difference (C − original)', res[k]['d'])):
        v = np.array(arr)
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"{k:<14} {tag:<26} {v.mean():+.3f}   [{lo:+.3f}, {hi:+.3f}]")
    print()

delta = point['score_ctx_lr_c'] - point['score_ctx_lr']
print("=== DECISION QUANTITY (rule fixed before execution) ===")
print(f"Point ΔAUROC, Context LR (Design C − original), full test set: "
      f"{delta:+.4f}   |Δ| = {abs(delta):.4f}")
if abs(delta) < 0.02:
    print("  Rule: |Δ| < 0.02 — sampling does not materially confound the comparison; no escalation.")
elif abs(delta) <= 0.05:
    print("  Rule: 0.02 ≤ |Δ| ≤ 0.05 — moderate confound; disclose as bounded limitation; no escalation.")
else:
    print("  Rule: |Δ| > 0.05 — substantial confound; report as a bounded limitation.")

print("\nDone.")
