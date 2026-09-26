"""
10_novelty_base_rates.py — novelty flag base rates (Sections 5.9.3.2 and 6.1.2)

Reports the novelty base rates side by side:
    P(src_novelty = 1 | attack)   vs   P(src_novelty = 1 | normal)
    P(dst_novelty = 1 | attack)   vs   P(dst_novelty = 1 | normal)

Source: test_features.pkl as produced by the corrected 3_features.py:
UNSCALED binary flags computed from full-training-period per-user
history. The flag means printed in the stage-3 capture (src 0.122/0.057,
dst 0.085/0.061) are these same conditional probabilities at 3 decimals;
this script adds the exact counts, more precision, and the rate ratio.

Read-only: loads test_features.pkl, writes nothing.

Both rates are reported side by side; no predictive conclusion is drawn
from the attack-side rate alone.
"""

import pandas as pd

print("Loading test_features.pkl ...")
test = pd.read_pickle('test_features.pkl')

y = test['label']
n_attack = int((y == 1).sum())
n_normal = int((y == 0).sum())
print(f"Test rows: {len(test):,}   attacks: {n_attack}   normals: {n_normal:,}")

print(f"\n{'flag':<14} {'P(=1 | attack)':>22} {'P(=1 | normal)':>26} {'ratio':>8}")
for flag in ['src_novelty', 'dst_novelty']:
    a_n = int(test.loc[y == 1, flag].sum())
    n_n = int(test.loc[y == 0, flag].sum())
    p_a = a_n / n_attack
    p_n = n_n / n_normal
    ratio = p_a / p_n if p_n > 0 else float('inf')
    print(f"{flag:<14} {a_n:>5}/{n_attack} = {p_a:8.4f} "
          f"{n_n:>13,}/{n_normal:,} = {p_n:6.4f} {ratio:>7.2f}x")

print("\nConsistency check: the stage-3 capture reported flag means "
      "src 0.122/0.057 and dst 0.085/0.061 (attacks/normal); the rates "
      "above must round to those values.")
print("\nBoth rates are reported side by side; no predictive conclusion "
      "is drawn from the attack-side rate alone.")
print("\nDone.")
