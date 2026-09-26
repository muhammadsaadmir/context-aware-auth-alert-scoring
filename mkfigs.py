"""mkfigs.py

Generates fig_auroc.pdf (thesis Figure 5.1) and fig_ablation.pdf
(thesis Figure 5.2). Every plotted value is PARSED from the capture
files of the corrected pipeline runs:

    05_bootstrap_output_batchC_rerun.txt  -> Figure 5.1 (bootstrap
        AUROC and 95% CI per comparator; Baseline 2b excluded as a
        diagnostic, see Section 5.6 / caption)
    04_evaluate_output_batchB_rerun.txt   -> Figure 5.2 (full-model
        and ablated AUROCs; deltas computed as FULL minus ABLATED,
        the convention of Table 5.4)

No result value is hard-coded. The EXPECTED_* dictionaries below are
guards only: every parsed value is asserted against them and the
script exits loudly on any mismatch, so it can never silently plot
stale data. The FORBIDDEN set holds the four pre-correction Figure
5.2 values; if any computed delta equals one of them the script
aborts.
"""

import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CAP_BOOTSTRAP = "05_bootstrap_output_batchC_rerun.txt"
CAP_EVALUATE  = "04_evaluate_output_batchB_rerun.txt"

# ---- guards (assertion targets only; plotted data comes from the captures) ----
EXPECTED_FIG1 = {
    "Baseline 1 (Severity-Only)":          (0.492, 0.485, 0.504),
    "Baseline 2 (Metadata-Only)":          (0.787, 0.762, 0.807),
    "Context-Aware (Equal Weights)":       (0.416, 0.352, 0.482),
    "Context-Aware (Logistic Regression)": (0.675, 0.619, 0.725),
}
EXPECTED_FULL_AUROC = 0.416
EXPECTED_DELTA = {                      # full minus ablated (Table 5.4)
    "historical_fail_propensity": +0.005,
    "time_deviation":             -0.315,
    "src_novelty":                +0.052,
    "dst_novelty":                +0.035,
}
FORBIDDEN = {-0.039, +0.294, -0.043, -0.025}   # pre-correction figure values
DIAGNOSTIC = "Baseline 2b (Metadata One-Hot)"  # excluded from Figure 5.1

def fail(msg):
    sys.exit("mkfigs: REFUSING TO PLOT - " + msg)

# ---- parse Figure 5.1 data from the stage-5 capture ----
ci_re = re.compile(
    r"^(?P<name>.+?)\s+AUROC = (?P<auroc>\d\.\d{3})\s+"
    r"\[95% CI: (?P<lo>\d\.\d{3})\s*\N{EN DASH}\s*(?P<hi>\d\.\d{3})\]\s*$")
fig1 = {}
for line in open(CAP_BOOTSTRAP, encoding="utf-8"):
    m = ci_re.match(line.rstrip("\n"))
    if m:
        fig1[m.group("name").strip()] = (
            float(m.group("auroc")), float(m.group("lo")), float(m.group("hi")))
if DIAGNOSTIC not in fig1:
    fail(f"diagnostic row '{DIAGNOSTIC}' not found in {CAP_BOOTSTRAP}")
fig1.pop(DIAGNOSTIC)                    # diagnostic, not a comparator
if set(fig1) != set(EXPECTED_FIG1):
    fail(f"comparator set mismatch in {CAP_BOOTSTRAP}: {sorted(fig1)}")
for name, vals in fig1.items():
    if vals != EXPECTED_FIG1[name]:
        fail(f"{name}: parsed {vals}, expected {EXPECTED_FIG1[name]}")

# ---- parse Figure 5.2 data from the stage-4 capture ----
full_re = re.compile(r"^Full model .*AUROC=(?P<a>\d\.\d{3})")
abl_re  = re.compile(r"^Without (?P<feat>\S+)\s+AUROC=(?P<a>\d\.\d{3})")
full_auroc, ablated = None, {}
for line in open(CAP_EVALUATE, encoding="utf-8"):
    m = full_re.match(line)
    if m:
        full_auroc = float(m.group("a"))
    m = abl_re.match(line)
    if m:
        ablated[m.group("feat")] = float(m.group("a"))
if full_auroc is None:
    fail(f"full-model AUROC not found in {CAP_EVALUATE}")
if full_auroc != EXPECTED_FULL_AUROC:
    fail(f"full-model AUROC parsed {full_auroc}, expected {EXPECTED_FULL_AUROC}")
if "fail_rate" in ablated:
    fail("obsolete feature name 'fail_rate' present in the capture")
if set(ablated) != set(EXPECTED_DELTA):
    fail(f"ablation feature set mismatch: {sorted(ablated)}")
delta = {f: round(full_auroc - a, 3) for f, a in ablated.items()}
for f, d in delta.items():
    if d != EXPECTED_DELTA[f]:
        fail(f"delta for {f}: computed {d:+.3f}, expected {EXPECTED_DELTA[f]:+.3f}")
    if d in FORBIDDEN:
        fail(f"delta for {f} equals a pre-correction value ({d:+.3f})")

print("parsed and asserted:")
for name in EXPECTED_FIG1:
    a, lo, hi = fig1[name]
    print(f"  {name}: {a:.3f} [{lo:.3f}, {hi:.3f}]")
for f in EXPECTED_DELTA:
    print(f"  delta {f}: {delta[f]:+.3f}  (full {full_auroc:.3f} - ablated {ablated[f]:.3f})")

# ---- styling (unchanged from the original script) ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": "#585A5B",
    "axes.linewidth": 0.8,
})
ORANGE = "#E84E0F"
GREY   = "#585A5B"
LGREY  = "#B9BBBC"

# ---------- Figure 5.1: AUROC with bootstrap 95% CIs (Table 5.5) ----------
order = ["Baseline 1 (Severity-Only)",
         "Baseline 2 (Metadata-Only)",
         "Context-Aware (Equal Weights)",
         "Context-Aware (Logistic Regression)"]
labels = ["Baseline 1\nSeverity-Only",
          "Baseline 2\nMetadata-Only",
          "Context-Aware\nEqual Weights",
          "Context-Aware\nLogistic Regression"]
auroc = [fig1[n][0] for n in order]
lo    = [fig1[n][1] for n in order]
hi    = [fig1[n][2] for n in order]
err = np.array([[a - l for a, l in zip(auroc, lo)],
                [h - a for a, h in zip(auroc, hi)]])
colors = [LGREY, ORANGE, LGREY, LGREY]

fig, ax = plt.subplots(figsize=(6.2, 3.1))
x = np.arange(len(labels))
ax.bar(x, auroc, width=0.55, color=colors, edgecolor=GREY, linewidth=0.7, zorder=3)
ax.errorbar(x, auroc, yerr=err, fmt="none", ecolor=GREY, elinewidth=1.0,
            capsize=4, capthick=1.0, zorder=4)
ax.axhline(0.5, color=GREY, linestyle="--", linewidth=0.9, zorder=2)
ax.text(-0.45, 0.508, "random expectation (0.500)",
        ha="left", va="bottom", fontsize=7.5, color=GREY)
for xi, v, h in zip(x, auroc, hi):
    ax.text(xi, max(v + 0.035, h + 0.012), f"{v:.3f}",
            ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("AUROC")
ax.set_ylim(0.30, 0.90)
ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("fig_auroc.pdf")

# ---------- Figure 5.2: Ablation delta AUROC, FULL minus ABLATED (Table 5.4) ----------
feats = ["historical_fail_propensity", "time_deviation", "src_novelty", "dst_novelty"]
dvals = [delta[f] for f in feats]
cols = [ORANGE if d < 0 else LGREY for d in dvals]   # orange marks the harming feature

fig, ax = plt.subplots(figsize=(6.2, 2.7))
y = np.arange(len(feats))[::-1]
ax.barh(y, dvals, height=0.5, color=cols, edgecolor=GREY, linewidth=0.7, zorder=3)
ax.axvline(0, color=GREY, linewidth=0.9, zorder=4)
for yi, d in zip(y, dvals):
    if d < -0.1:                      # long negative bar: label inside, white
        ax.text(d + 0.010, yi, f"{d:+.3f}", va="center", ha="left",
                fontsize=8.5, color="white")
    else:
        off = 0.008 if d > 0 else -0.008
        ha = "left" if d > 0 else "right"
        ax.text(d + off, yi, f"{d:+.3f}", va="center", ha=ha, fontsize=8.5)
ax.set_yticks(y)
ax.set_yticklabels(feats, fontfamily="monospace", fontsize=9)
ax.set_xlabel(r"$\Delta$AUROC, full model minus ablated model")
ax.set_xlim(-0.36, 0.12)
ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig("fig_ablation.pdf")
print("figures written: fig_auroc.pdf, fig_ablation.pdf")
print("copy as: figures/fig-4-1.pdf and figures/fig-4-2.pdf in the LaTeX tree")
