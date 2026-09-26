"""
13_lr_ablation.py — drop-one refit ablation of the context-aware logistic regression

Table 5.4 attributes each feature's contribution on the equal-weight scorer,
where removal is clean (the ablated score is the full score minus one term).
No ablation was run on the learned-weight scorer, because a removal there
means a refit. This script performs that refit: it rebuilds the reference/fit
partition and the fit-window features exactly as 4_evaluate.py does, refits
the full four-feature regression and asserts that its coefficients and test
AUROC reproduce the stage-4 capture, then fits the regression four more
times, each without one feature, and scores the unchanged test set with each
three-feature model. Removal effects are reported under the full-minus-ablated
convention of Table 5.4, and the three learned coefficients of every ablated
model are printed so that a sign change on removal is visible.

Inputs: train.pkl, redteam.txt (labels), test_features.pkl (raw test features
from 3_features.py). The scaler is fitted on the fit window and applied to the
test features, as in stage 4; each ablated model uses the same scaled columns
minus one, so the scaling is identical across the five fits.

Deterministic: LogisticRegression(max_iter=1000, class_weight='balanced') with
the default lbfgs solver, no random_state, as in 4_evaluate.py. Reads only;
writes nothing. Post-protocol; does not enter the adjudication of either
research question. Generated using AI (Claude, Anthropic) in the correction phase.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import circmean

T_REF = 1002240
CONTEXT_COLS = ['historical_fail_propensity', 'time_deviation',
                'src_novelty', 'dst_novelty']

# Values of record (04_evaluate_output_batchB_rerun.txt), asserted below
EXPECTED_COEF = {'historical_fail_propensity': -17.142, 'time_deviation': -4.399,
                 'src_novelty': 4.443, 'dst_novelty': 4.064}
EXPECTED_AUROC = 0.676
EXPECTED_MEDIAN_RANK = 10_285_800

print("Loading data ...")
train = pd.read_pickle('train.pkl')
test = pd.read_pickle('test_features.pkl')
test = test[['label'] + CONTEXT_COLS].copy()
y_test = test['label'].astype(int)
print(f"Train {len(train):,} rows | Test {len(test):,} rows, {int(y_test.sum())} attacks")

train['hour'] = (train['time'] % 86400) // 3600
redteam = pd.read_csv('redteam.txt', header=None,
                      names=['time', 'user', 'src_computer', 'dst_computer'])
attack_keys = set(zip(redteam['time'], redteam['src_computer'], redteam['dst_computer']))
train['label'] = train.apply(
    lambda r: 1 if (r['time'], r['src_computer'], r['dst_computer']) in attack_keys else 0, axis=1)

# ── reference/fit partition and fit-window features, as in 4_evaluate.py ──
reference = train[train['time'] < T_REF]
fit = train[train['time'] >= T_REF].copy()
print(f"Reference window: {len(reference):,} rows   Fit window: {len(fit):,} rows   "
      f"Attacks in fit window: {int(fit['label'].sum())}")

user_total = reference.groupby('src_user').size().rename('total_events')
user_fail = reference[reference['success'] == 'Fail'].groupby('src_user').size().rename('fail_events')
hist_fail_lookup = (user_fail / user_total).rename('historical_fail_propensity').fillna(0)
mean_hour_lookup = (reference.groupby('src_user')['hour']
                    .agg(lambda h: circmean(h, high=24, low=0)).rename('mean_hour'))
known_src_lookup = reference.groupby('src_user')['src_computer'].apply(set).rename('known_src')
known_dst_lookup = reference.groupby('src_user')['dst_computer'].apply(set).rename('known_dst')

fit = fit.join(hist_fail_lookup, on='src_user')
fit = fit.join(mean_hour_lookup, on='src_user')
fit = fit.join(known_src_lookup, on='src_user')
fit = fit.join(known_dst_lookup, on='src_user')
fit['historical_fail_propensity'] = fit['historical_fail_propensity'].fillna(0)
fit['mean_hour'] = fit['mean_hour'].fillna(12)
hour_diff = (fit['hour'] - fit['mean_hour']).abs()
fit['time_deviation'] = np.minimum(hour_diff, 24 - hour_diff)
fit['src_novelty'] = fit.apply(
    lambda r: 1 if isinstance(r['known_src'], set) and r['src_computer'] not in r['known_src'] else 0, axis=1)
fit['dst_novelty'] = fit.apply(
    lambda r: 1 if isinstance(r['known_dst'], set) and r['dst_computer'] not in r['known_dst'] else 0, axis=1)
y_fit = fit['label'].astype(int)
del train, reference

scaler = MinMaxScaler()
X_fit = pd.DataFrame(scaler.fit_transform(fit[CONTEXT_COLS].fillna(0)), columns=CONTEXT_COLS)
X_test = pd.DataFrame(scaler.transform(test[CONTEXT_COLS].fillna(0)), columns=CONTEXT_COLS)
del fit, test


def median_attack_rank(scores):
    ranks = pd.Series(scores).rank(ascending=False)
    return float(ranks[y_test.to_numpy() == 1].median())


def fit_and_score(cols):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_fit[cols], y_fit)
    scores = model.predict_proba(X_test[cols])[:, 1]
    return model, float(roc_auc_score(y_test, scores)), median_attack_rank(scores)


# ── full model: reproduction check against the stage-4 capture ──
print("\nRefitting the full four-feature context regression ...")
full_model, full_auroc, full_rank = fit_and_score(CONTEXT_COLS)
print("Learned coefficients (should reproduce the stage-4 capture):")
for name, coef in zip(CONTEXT_COLS, full_model.coef_[0]):
    print(f"  {name:<28} {coef:+.3f}")
    assert abs(coef - EXPECTED_COEF[name]) < 5e-4, f"{name}: {coef:+.3f} vs {EXPECTED_COEF[name]:+.3f}"
print(f"Full model  AUROC={full_auroc:.3f}  Median attack rank={full_rank:,.0f}")
assert abs(full_auroc - EXPECTED_AUROC) < 5e-4, f"AUROC {full_auroc:.3f} vs {EXPECTED_AUROC:.3f}"
assert round(full_rank) == EXPECTED_MEDIAN_RANK, f"median rank {full_rank:,.0f} vs {EXPECTED_MEDIAN_RANK:,}"
print("Reproduction check passed (coefficients, AUROC and median attack rank equal the capture).")

# ── drop-one refits ──
print("\n=== DROP-ONE REFIT ABLATION (context-aware logistic regression) ===")
print("ΔAUROC = full model − model refitted without the feature (full minus ablated,")
print("as in Table 5.4); Δ median rank = full-model rank − ablated rank.\n")
for drop in CONTEXT_COLS:
    cols = [c for c in CONTEXT_COLS if c != drop]
    model, auroc, rank = fit_and_score(cols)
    coefs = "  ".join(f"{c}={w:+.3f}" for c, w in zip(cols, model.coef_[0]))
    print(f"Without {drop:<28} AUROC={auroc:.3f} (Δ{full_auroc - auroc:+.3f})   "
          f"Median rank={rank:,.0f} (Δ{full_rank - rank:+,.0f})")
    print(f"    refitted coefficients: {coefs}")

print("\nEach ablated model is a fresh fit on the three remaining scaled features;")
print("coefficients are conditional on the features present and may change sign.")
print("Done.")
