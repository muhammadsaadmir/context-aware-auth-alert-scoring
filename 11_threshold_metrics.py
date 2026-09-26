"""
11_threshold_metrics.py — threshold metrics named in the proposal (Recall@k, F1, FPR)

The proposal listed Precision@k, Recall@k, F1, the false-positive rate and AUROC
as evaluation metrics. Stage 4 printed Precision@k and AUROC; this script
computes the remaining three after the fact, from the scores stage 4 saved in
test_scored.pkl, so that the deviation register in Appendix B.2 can state what
those metrics are rather than only why they were not computed.

Every threshold metric needs a decision rule. Two families are reported:

  (1) Top-k rule: the k highest-scoring events are alerted, k = 20, 100, 1000,
      the same depths and the same tie handling (Series.nlargest) as the
      Precision@k of stage 4. Precision@k reproduced here must equal the
      stage-4 capture; the script asserts it.
  (2) Natural decision cut of each scorer: score >= 0.5 for the three logistic
      regressions (a posterior of at least one half), score = 1 for Baseline 1
      (every failed login alerted). The equal-weight score is a mean of four
      scaled features in [0, 1] with no probabilistic reading, so it has no
      natural cut and only the top-k rule applies to it.

Definitions (positives = red-team rows, P = 82; negatives N = 39,119,426):
  Recall@k  = TP / P          F1 = 2 TP / (2 TP + FP + FN)
  Precision = TP / (TP + FP)  FPR = FP / N

Deterministic: no random draw. Reads test_scored.pkl only; writes nothing.
Generated using AI (Claude, Anthropic) in the correction phase.
"""
import gc
import numpy as np
import pandas as pd

METHODS = [
    ("Baseline 1 (Severity-Only)",          "score_b1"),
    ("Baseline 2 (Metadata-Only)",          "score_b2"),
    ("Baseline 2b (Metadata One-Hot)",      "score_b2_onehot"),
    ("Context-Aware (Equal Weights)",       "score_ctx_equal"),
    ("Context-Aware (Logistic Regression)", "score_ctx_lr"),
]
KS = [20, 100, 1000]

# Precision@k of record (04_evaluate_output_batchB_rerun.txt), asserted below
EXPECTED_P_AT_K = {
    "score_b1":        {20: 0.000, 100: 0.000, 1000: 0.000},
    "score_b2":        {20: 0.000, 100: 0.000, 1000: 0.000},
    "score_b2_onehot": {20: 0.650, 100: 0.620, 1000: 0.080},
    "score_ctx_equal": {20: 0.000, 100: 0.000, 1000: 0.000},
    "score_ctx_lr":    {20: 0.000, 100: 0.000, 1000: 0.000},
}

print("Loading test_scored.pkl ...")
test = pd.read_pickle("test_scored.pkl")
cols = ["label"] + [c for _, c in METHODS]
test = test[cols].copy()
gc.collect()
labels = test["label"].astype(int)
P = int(labels.sum())
N = int(len(labels) - P)
print(f"Test events: {len(labels):,}   Attacks (P): {P}   Normal (N): {N:,}\n")


def confusion(alert_mask):
    tp = int((alert_mask & (labels == 1)).sum())
    fp = int((alert_mask & (labels == 0)).sum())
    fn = P - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / P
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    fpr = fp / N
    return tp, fp, fn, prec, rec, f1, fpr


def line(rule, tp, fp, fn, prec, rec, f1, fpr):
    print(f"  {rule:<22} TP={tp:>3}  FP={fp:>10,}  FN={fn:>3}  "
          f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  FPR={fpr:.6f}")


print("=== TOP-k RULE (k highest-scoring events alerted) ===")
for name, col in METHODS:
    print(name)
    scores = test[col]
    for k in KS:
        top = scores.nlargest(k).index          # same call as stage 4
        mask = pd.Series(False, index=test.index)
        mask.loc[top] = True
        tp, fp, fn, prec, rec, f1, fpr = confusion(mask)
        assert abs(prec - EXPECTED_P_AT_K[col][k]) < 5e-4, \
            f"Precision@{k} for {col} = {prec:.3f}, expected {EXPECTED_P_AT_K[col][k]:.3f}"
        line(f"k = {k:,}", tp, fp, fn, prec, rec, f1, fpr)
    print()

print("=== NATURAL DECISION CUT OF EACH SCORER ===")
for name, col in METHODS:
    scores = test[col]
    if col == "score_b1":
        rule, mask = "score = 1 (all fails)", scores >= 1.0
    elif col == "score_ctx_equal":
        print(name)
        print("  no natural cut: the equal-weight score is not a probability\n")
        continue
    else:
        rule, mask = "p >= 0.5", scores >= 0.5
    print(name)
    tp, fp, fn, prec, rec, f1, fpr = confusion(mask)
    print(f"  alerted events: {int(mask.sum()):,}")
    line(rule, tp, fp, fn, prec, rec, f1, fpr)
    print()

print("Precision@k reproduced from test_scored.pkl equals the stage-4 capture "
      "for all five methods and all three depths (asserted).")
print("Done.")
