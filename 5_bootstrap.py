import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

print("Loading scored test data...")
test = pd.read_pickle('test_scored.pkl')

y_test = test['label']
attacks = test[y_test == 1]
normals = test[y_test == 0]

print(f"Attacks: {len(attacks)}  Normals: {len(normals):,}")

# Fixed random sample of normal events, used for computational feasibility.
# Bootstrapping all 39M negatives 1000 times is impractical; this is a
# standard adaptation for highly imbalanced AUROC bootstrap analysis.
NEG_SAMPLE_SIZE = 200_000
rng = np.random.default_rng(42)
neg_sample_idx = rng.choice(normals.index, size=NEG_SAMPLE_SIZE, replace=False)
neg_sample = normals.loc[neg_sample_idx]

print(f"Using a fixed random sample of {NEG_SAMPLE_SIZE:,} normal events for bootstrap\n")

methods = {
    'Baseline 1 (Severity-Only)':          'score_b1',
    'Baseline 2 (Metadata-Only)':          'score_b2',
    'Baseline 2b (Metadata One-Hot)':      'score_b2_onehot',
    'Context-Aware (Equal Weights)':       'score_ctx_equal',
    'Context-Aware (Logistic Regression)': 'score_ctx_lr',
}

N_BOOTSTRAP = 1000
results = {name: [] for name in methods}

# RQ2 (P0-4): paired ablation deltas on the equal-weight score.
# The equal weight per feature is 0.25 (as in 4_evaluate.py), so the score
# without a feature is score_ctx_equal − 0.25 × that feature. Each delta is
# computed on the SAME resample as the full model, giving a paired bootstrap.
W_EQUAL = 0.25
ABLATE_FEATURES = ['historical_fail_propensity', 'time_deviation',
                   'src_novelty', 'dst_novelty']
rq2_deltas = {f: [] for f in ABLATE_FEATURES}

print(f"Running {N_BOOTSTRAP} paired bootstrap resamples...")
for i in range(N_BOOTSTRAP):
    boot_attacks = attacks.sample(n=len(attacks), replace=True, random_state=i)
    boot_negs    = neg_sample.sample(n=len(neg_sample), replace=True, random_state=i + 10000)
    boot_set     = pd.concat([boot_attacks, boot_negs])
    boot_labels  = boot_set['label']

    for name, col in methods.items():
        try:
            auroc = roc_auc_score(boot_labels, boot_set[col])
        except ValueError:
            auroc = np.nan
        results[name].append(auroc)

    # RQ2: paired delta per ablated feature on this same resample
    full_auroc = roc_auc_score(boot_labels, boot_set['score_ctx_equal'])
    for f in ABLATE_FEATURES:
        ablated = boot_set['score_ctx_equal'] - W_EQUAL * boot_set[f]
        rq2_deltas[f].append(full_auroc - roc_auc_score(boot_labels, ablated))

    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{N_BOOTSTRAP} done")

print("\n=== BOOTSTRAP AUROC — 95% CONFIDENCE INTERVALS ===\n")
auroc_arrays = {}
for name in methods:
    vals = np.array(results[name])
    vals = vals[~np.isnan(vals)]
    auroc_arrays[name] = vals
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"{name:<38} AUROC = {vals.mean():.3f}   [95% CI: {lo:.3f} – {hi:.3f}]")

# ── RQ1: Context-aware vs each baseline, tested separately ─────────────────
ctx_best = auroc_arrays['Context-Aware (Logistic Regression)']
for base_name in ['Baseline 2 (Metadata-Only)', 'Baseline 2b (Metadata One-Hot)']:
    print("\n=== RQ1 SIGNIFICANCE TEST ===")
    print(f"Context-Aware (Logistic Regression) vs {base_name}\n")

    diff = ctx_best - auroc_arrays[base_name]
    diff_lo, diff_hi = np.percentile(diff, [2.5, 97.5])

    print(f"AUROC difference (Context-Aware − {base_name}):")
    print(f"  Mean:    {diff.mean():+.3f}")
    print(f"  95% CI:  [{diff_lo:+.3f}, {diff_hi:+.3f}]")

    if diff_lo > 0:
        print("\n  --> CI excludes zero, positive: context-aware SIGNIFICANTLY outperforms this baseline.")
    elif diff_hi < 0:
        print("\n  --> CI excludes zero, negative: context-aware SIGNIFICANTLY underperforms this baseline.")
    else:
        print("\n  --> CI includes zero: difference is NOT statistically significant at this sample size.")

# ── RQ2: paired-bootstrap CI for each ablated feature (equal-weight score) ──
print("\n=== RQ2 ABLATION — PAIRED BOOTSTRAP 95% CONFIDENCE INTERVALS ===")
print("ΔAUROC = full equal-weight model − model without the feature")
print("(positive Δ: the feature helps; CI excluding zero: the change is")
print("statistically distinguishable)\n")
for f in ABLATE_FEATURES:
    d = np.array(rq2_deltas[f])
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    verdict = "excludes zero" if (lo > 0 or hi < 0) else "includes zero"
    print(f"Without {f:<28} ΔAUROC = {d.mean():+.3f}   [95% CI: {lo:+.3f}, {hi:+.3f}]   CI {verdict}")

print("\nDone.")