"""
12_feature_auroc.py — single-feature AUROC of the four raw context features

Chapter 5 attributes each feature's contribution by removing it from the
equal-weight score (Table 5.4) and reads each feature's direction from its
class means (Table 5.1). A mean is not a rank statistic: a feature can have a
lower mean among attacks and still rank most attacks above most normal
events when the normal population sits at a point mass. This script reports
the quantity the direction question needs, the AUROC of each raw feature used
as a score on its own over the full test set, so that "attacks rank higher"
or "attacks rank lower" is measured rather than inferred. It adds the AUROC
of the two novelty flags summed at equal weight (the novelty-only scorer
named in Section 6.3) and, for the propensity feature, the share of each
class sitting at exactly zero, which is what separates its mean from its rank
behaviour.

For a binary flag the AUROC has the closed form 0.5 + 0.5 (p_attack - p_normal)
with ties counted at one half; the script computes it from the same counts as
10_novelty_base_rates.py and asserts that roc_auc_score agrees, which ties
this listing to the base rates of Figure A.18. The feature means are asserted
against the stage-3 capture (Figure A.6) before anything else is printed.

Scaling in stage 4 is a monotone map within each feature, so the raw values
give the same AUROC as the scaled values the scorers consume. Deterministic:
no random draw. Reads test_features.pkl only; writes nothing.
Generated using AI (Claude, Anthropic) in the correction phase.
"""
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FEATURES = ["historical_fail_propensity", "time_deviation",
            "src_novelty", "dst_novelty"]

# Feature means of record (03_features_output_batchA_rerun.txt), asserted below
EXPECTED_MEANS = {
    "historical_fail_propensity": (0.012, 0.024),
    "time_deviation":             (3.162, 5.364),
    "src_novelty":                (0.122, 0.057),
    "dst_novelty":                (0.085, 0.061),
}

print("Loading test_features.pkl ...")
test = pd.read_pickle("test_features.pkl")
test = test[["label"] + FEATURES].copy()
gc.collect()
labels = test["label"].astype(int).to_numpy()
P = int(labels.sum())
N = int(len(labels) - P)
print(f"Test events: {len(labels):,}   Attacks: {P}   Normal: {N:,}\n")

is_attack = labels == 1
for f in FEATURES:
    a_mean = float(test.loc[is_attack, f].mean())
    n_mean = float(test.loc[~is_attack, f].mean())
    ea, en = EXPECTED_MEANS[f]
    assert round(a_mean, 3) == ea and round(n_mean, 3) == en, \
        f"{f}: means {a_mean:.3f}/{n_mean:.3f}, stage-3 capture {ea:.3f}/{en:.3f}"
print("Feature means agree with the stage-3 capture for all four features (asserted).\n")

print("=== SINGLE-FEATURE AUROC (each raw feature used as the score on its own) ===")
print("AUROC = P(attack value > normal value) + 0.5 P(equal); above 0.500 means")
print("attacks rank higher than normal events on that feature, below means lower.\n")
print(f"{'feature':<28} {'AUROC':>7}   {'attack mean':>11} {'normal mean':>11}   direction at rank level")
auroc = {}
for f in FEATURES:
    auroc[f] = float(roc_auc_score(labels, test[f].to_numpy()))
    a_mean = float(test.loc[is_attack, f].mean())
    n_mean = float(test.loc[~is_attack, f].mean())
    if auroc[f] > 0.5:
        direction = "attacks rank higher"
    elif auroc[f] < 0.5:
        direction = "attacks rank lower"
    else:
        direction = "no separation"
    print(f"{f:<28} {auroc[f]:7.4f}   {a_mean:11.3f} {n_mean:11.3f}   {direction}")

print("\n=== BINARY FLAGS: closed form 0.5 + 0.5 (p_attack - p_normal) ===")
for f in ["src_novelty", "dst_novelty"]:
    p_a = float(test.loc[is_attack, f].mean())
    p_n = float(test.loc[~is_attack, f].mean())
    closed = 0.5 + 0.5 * (p_a - p_n)
    assert abs(closed - auroc[f]) < 1e-6, \
        f"{f}: closed form {closed:.6f} differs from roc_auc_score {auroc[f]:.6f}"
    print(f"{f:<14} p_attack={p_a:.4f}  p_normal={p_n:.4f}  "
          f"closed form={closed:.4f}  roc_auc_score={auroc[f]:.4f}  (agree, asserted)")

print("\n=== NOVELTY-ONLY SCORER (0.5 x src_novelty + 0.5 x dst_novelty) ===")
novelty_only = 0.5 * test["src_novelty"].to_numpy() + 0.5 * test["dst_novelty"].to_numpy()
auroc_nov = float(roc_auc_score(labels, novelty_only))
print(f"AUROC = {auroc_nov:.4f}   (Section 6.3 names this as the most direct next experiment)")

print("\n=== historical_fail_propensity: mean against rank ===")
hfp = test["historical_fail_propensity"].to_numpy()
zero_a = float((hfp[is_attack] == 0).mean())
zero_n = float((hfp[~is_attack] == 0).mean())
med_a = float(np.median(hfp[is_attack]))
med_n = float(np.median(hfp[~is_attack]))
mean_a = float(hfp[is_attack].mean())
mean_n = float(hfp[~is_attack].mean())
print(f"share of events with propensity exactly 0:  attacks {zero_a:.4f}   normal {zero_n:.4f}")
print(f"median propensity:                          attacks {med_a:.4f}   normal {med_n:.4f}")
print(f"mean propensity (Table 5.1):                attacks {mean_a:.3f}   normal {mean_n:.3f}")
if auroc["historical_fail_propensity"] > 0.5 and mean_a < mean_n:
    print("The mean is lower for attacks and the AUROC is above 0.5: the normal side")
    print("carries a heavier upper tail while more of it sits at zero. A mean sees the")
    print("tail; a rank statistic sees the point mass. The two do not disagree, they")
    print("measure different things.")
elif auroc["historical_fail_propensity"] < 0.5 and mean_a < mean_n:
    print("The mean and the AUROC agree: attacks sit lower on this feature at the")
    print("rank level as well as on average.")
else:
    print("Mean and rank read in the same direction on this feature.")

print("\nSingle-feature AUROC is invariant to the monotone min-max scaling of stage 4,")
print("so these values apply to the scaled features the scorers consume.")
print("Done.")
