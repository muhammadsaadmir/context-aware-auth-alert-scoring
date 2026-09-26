# README — Reproducibility Artefact
# Context-Aware Risk Scoring for Prioritizing
# Authentication Alerts in Security Operations Centers
#
# Author:        Muhammad Saad Mir
# Matriculation: 100001795
# Programme:     M.Sc. Computer Science focus Cyber Security
# University:    SRH University of Applied Sciences Heidelberg,
#                Campus Leipzig
# Supervisors:   Prof. Dr. Klaus Dieter Schwarz (first),
#                Franziska Schwarz (second)
# Submitted:     September 2026
# ─────────────────────────────────────────────────────────────────

## OVERVIEW

This artefact contains all code, configuration and documentation required
to reproduce the experimental results reported in Chapter 5 of the thesis,
starting from the publicly available LANL dataset. The five pipeline scripts
(1 to 5) produce every comparator result, the ablation and the bootstrap
intervals; the eight diagnostic scripts (6 to 13) produce the Baseline 2
mechanism diagnostics, the sampling-sensitivity analysis, the novelty
base rates, the threshold metrics, the single-feature AUROC values and the
refit ablation of the context logistic regression. Every stage is deterministic or seeded, so a re-execution on the
same working file reproduces the reported values exactly.

The console output of each executed run is included as a capture file and is
reproduced verbatim in Appendix A of the thesis. Appendix B.1 lists the
contents of this artefact and Appendix B.2 records every deviation from the
pre-registered proposal.

---

## CONTENTS OF THIS ARTEFACT

| File | Purpose |
|---|---|
| README.md | This file: environment, execution order, expected outputs |
| preprocessing.sh | Builds auth_working.txt from the raw LANL files (grep, awk, cat, sort) |
| 1_load_check.py | Stage 1: loads the working file and checks counts, schema and time range |
| 2_split.py | Stage 2: temporal split at second 1,252,799; writes train.pkl and test.pkl |
| 3_features.py | Stage 3: computes the four context features from the training period and applies them to the test set; writes test_features.pkl |
| 4_evaluate.py | Stage 4: Baselines 1, 2 and 2b, both context-aware scorers, Precision@k, median attack rank, single-feature ablation; writes test_scored.pkl |
| 5_bootstrap.py | Stage 5: 1,000 paired bootstrap resamples, AUROC confidence intervals, both RQ1 significance tests, RQ2 paired ablation intervals |
| 6 baseline2 mechanism.py | Diagnostics, Experiments A to C: Baseline 2 coefficients, twenty permuted encodings, identity-removed refit |
| 7_b2b_mechanism.py | Diagnostics, Experiments D and E: identity overlap across the split, Baseline 2b coefficient and purity census; writes b2b_mechanism_top20.csv |
| 8_sampling_step0.py | Measures the two sampling regimes of the working file |
| 9_sensitivityC.py | Density-equalised sensitivity analysis (Design C) with its paired bootstrap; writes test_scored_c.pkl |
| 10_novelty_base_rates.py | Attack-side and normal-side base rates of both novelty flags |
| 11_threshold_metrics.py | Recall@k, F1 and false-positive rate at the top-k and natural decision cuts, from the scores saved by stage 4 |
| 12_feature_auroc.py | AUROC of each raw context feature used as a score on its own, and of the novelty-only scorer, from the features saved by stage 3 |
| 13_lr_ablation.py | Drop-one refit ablation of the context logistic regression: reproduces the stage-4 fit, then refits without each feature and scores the test set |
| check_dupes.py | Diagnostic: confirms that no labelled attack row is an exact duplicate of another in either partition |
| mkfigs.py | Regenerates Figures 5.1 and 5.2 by parsing the stage-4 and stage-5 capture files; every parsed value is asserted against the reported results |
| seeds.txt | Every source of randomness in the pipeline, with the seed fixed for each |
| requirements.txt | Pinned library versions (pip freeze, Python 3.13) |
| 03_features_output_batchA_rerun.txt | Capture of the stage-3 run (Figure A.6) |
| 04_evaluate_output_batchB_rerun.txt | Capture of the stage-4 run (Figures A.7 and A.8) |
| 05_bootstrap_output_batchC_rerun.txt | Capture of the stage-5 run (Figures A.9 and A.10) |
| 06_baseline2_output.txt | Capture of script 6 (Figures A.11 and A.12) |
| 07_b2b_mechanism_output.txt | Capture of script 7 (Figures A.13 and A.14) |
| 08_sampling_step0_output.txt | Capture of script 8 (Figure A.15) |
| 09_sensitivityC_output.txt | Capture of script 9 (Figures A.16 and A.17) |
| 10_novelty_base_rates_output.txt | Capture of script 10 (Figure A.18) |
| 11_threshold_metrics_output.txt | Capture of script 11 (Figure A.19) |
| 12_feature_auroc_output.txt | Capture of script 12 (Figure A.20) |
| 13_lr_ablation_output.txt | Capture of script 13 (Figure A.21) |
| b2b_mechanism_top20.csv | Top-twenty coefficient listing of Baseline 2b, written by script 7 (Table 5.7) |
| candidates_raw.csv | All 1,634 records retrieved for the literature review, with the source database and query for each |
| candidates_triaged.csv | The 1,615 deduplicated records with triage label and reason |

---

## DATASET

**Name:** LANL Comprehensive Multi-Source Cyber-Security Events
**Author:** Kent, A. D. (2015)
**DOI:** 10.17021/1179829
**Download:** https://csr.lanl.gov/data/cyber1/

The dataset as a whole records 1,648,275,307 events for 12,425 users,
17,684 computers and 62,974 processes over 58 days. Only two of its files
are used here:

| File | Compressed | Uncompressed | Content |
|---|---|---|---|
| auth.txt | ~7 GB | 73.4 GB | Authentication events, about 1.05 billion rows (the 1-in-1,000 systematic sample in preprocessing.sh returns 1,051,430 rows) |
| redteam.txt | tiny | 22 KB | 749 red-team records |

```bash
wget https://csr.lanl.gov/data/cyber1/auth.txt.gz
wget https://csr.lanl.gov/data/cyber1/redteam.txt.gz
gunzip auth.txt.gz
gunzip redteam.txt.gz
```

auth.txt cannot be loaded into memory. preprocessing.sh reduces it to the
51,031,056-row working file that every Python script reads.

---

## SYSTEM REQUIREMENTS

- **Machine:** the reported results were produced on an Apple MacBook Air
  (M3, 8 GB RAM) running macOS Tahoe 26.2.
- **Python:** 3.13 (the interpreter that produced the reported results).
  The pinned versions in requirements.txt need Python 3.11 or later.
- **Libraries:** pandas, numpy, scikit-learn, scipy; matplotlib for
  mkfigs.py only. Exact versions in requirements.txt.
- **Memory:** the reported results were produced on a machine with 8 GB of
  RAM. Stages 3, 4 and 9 hold the 39.1-million-row test frame in memory;
  more memory shortens their runtime but does not change any value.
- **Disk:** roughly 100 GB free during preprocessing (auth.txt.gz 7 GB,
  auth.txt 73.4 GB, auth_attacks.txt about 3 GB, auth_normal_sample.txt
  about 70 MB, auth_working.txt 3.1 GB, temporary space for sort, and the
  pickle files listed below). Deleting auth.txt.gz after decompression and
  the two intermediates after auth_working.txt exists brings the peak to
  about 88 GB.
- **Time:** 30 to 60 minutes for preprocessing.sh, then 30 to 50 minutes
  for stages 1 to 5. Script 6 refits Baseline 2 once per permutation
  (twenty refits); script 7 takes about one stage-4 run; script 9 repeats
  the feature construction and the bootstrap of stages 3 to 5; script 13
  repeats the fit-window feature construction of stage 4 and fits five
  regressions; scripts 8, 10, 11 and 12 only read the pickles and count or
  rank.

---

## REPRODUCTION STEPS

Run every script from the folder that contains auth_working.txt and
redteam.txt.

### Step 1 — Install the dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Build the working file (skip if auth_working.txt exists)

```bash
chmod +x preprocessing.sh
./preprocessing.sh
```

1. Extracts every authentication event involving a red-team account with
   grep: auth_attacks.txt, 49,979,626 rows.
2. Takes a systematic 1-in-1,000 sample of auth.txt with awk:
   auth_normal_sample.txt, 1,051,430 rows.
3. Concatenates both and sorts by timestamp: auth_working.txt,
   51,031,056 rows, 3.1 GB. The script prints the observed row count
   against the expected 51,031,056 and warns if they differ.

The two passes overlap: the sample is drawn from the whole of auth.txt,
including rows the first pass already extracted, and the two outputs are
concatenated without de-duplication. This is why the two counts sum
exactly to 51,031,056 and why the working file holds 50,280 distinct lines
that occur more than once (0.099 percent). Section 4.4 of the thesis
documents this; check_dupes.py confirms that no labelled attack row is
affected.

### Step 3 — Stage 1: load and integrity check

```bash
python3 1_load_check.py
```

Expected: 51,031,056 authentication events, 749 red-team records, success
values ['Success' 'Fail'], time range 1 to 5011199.

### Step 4 — Stage 2: temporal split

```bash
python3 2_split.py
```

Expected:
```
Split time: 1252799
Training events: 11,911,548
Test events:     39,119,508
Attacks found in test set: 82
Normal events in test set: 39,119,426
Saved train.pkl and test.pkl
```

The split is deterministic: floor(5,011,199 x 0.25) = 1,252,799, applied by
timestamp comparison. redteam.txt holds 749 records that reduce to 713
distinct (time, src_computer, dst_computer) keys, 633 in the training period
and 80 in the test period. Of the 633 training keys, 609 are present in the
working file and match 610 rows; all 80 test keys are present and match 82
rows, because one tuple carries two accounts and matches three rows.
Section 4.4 of the thesis sets this out in full.

### Step 5 — Stage 3: feature engineering

```bash
python3 3_features.py
```

Expected output (03_features_output_batchA_rerun.txt):
```
Loading training and test data...
Train: 11,911,548  Test: 39,119,508
Computing features from training period only...
Applying features to test set...

Feature means — ATTACKS vs NORMAL:
  historical_fail_propensity attacks=0.012  normal=0.024
  time_deviation       attacks=3.162  normal=5.364
  src_novelty          attacks=0.122  normal=0.057
  dst_novelty          attacks=0.085  normal=0.061

Saved test_features.pkl
Feature engineering complete.
```

The four features are computed from the full training period only. The
mean login hour is the circular mean on a 24-hour cycle
(scipy.stats.circmean, high=24) and time_deviation is the circular distance
to it, at most 12 hours. A user absent from the training period receives
historical_fail_propensity 0, mean hour 12 and both novelty flags 0. No
scaling happens in this stage.

### Step 6 — Stage 4: scoring, evaluation and ablation

```bash
python3 4_evaluate.py
```

Expected output (04_evaluate_output_batchB_rerun.txt):
```
Loading data...
Test events: 39,119,508  Attacks: 82
Training metadata-only baseline...
Training metadata-only baseline (one-hot)...
Building context features for training set (three-way partition)...
Reference window: 9,222,877 rows   Fit window: 2,688,671 rows   Attacks in fit window: 299
Training context-aware logistic regression (4 context features only)...
Learned context feature weights (logistic regression coefficients):
  historical_fail_propensity -17.142
  time_deviation     -4.399
  src_novelty        +4.443
  dst_novelty        +4.064

=== RESULTS ===

Baseline 1 (Severity-Only)
  AUROC:              0.492
  Precision@20:       0.000
  Precision@100:      0.000
  Precision@1000:     0.000
  Median attack rank: 20,118,464 out of 39,119,508
  (random expectation = 19,559,754; lower is better)

Baseline 2 (Metadata-Only)
  AUROC:              0.787
  Precision@20:       0.000
  Precision@100:      0.000
  Precision@1000:     0.000
  Median attack rank: 7,795,882 out of 39,119,508
  (random expectation = 19,559,754; lower is better)

Baseline 2b (Metadata One-Hot)
  AUROC:              1.000
  Precision@20:       0.650
  Precision@100:      0.620
  Precision@1000:     0.080
  Median attack rank: 55 out of 39,119,508
  (random expectation = 19,559,754; lower is better)

Context-Aware (Equal Weights)
  AUROC:              0.416
  Precision@20:       0.000
  Precision@100:      0.000
  Precision@1000:     0.000
  Median attack rank: 26,419,052 out of 39,119,508
  (random expectation = 19,559,754; lower is better)

Context-Aware (Logistic Regression)
  AUROC:              0.676
  Precision@20:       0.000
  Precision@100:      0.000
  Precision@1000:     0.000
  Median attack rank: 10,285,800 out of 39,119,508
  (random expectation = 19,559,754; lower is better)

=== ABLATION STUDY (equal-weight context score) ===
Full model — AUROC=0.416  Median rank=26,419,052 out of 39,119,508

Without historical_fail_propensity AUROC=0.411 (Δ-0.005)   Median rank=26,304,230 (Δ+114,822)
Without time_deviation     AUROC=0.731 (Δ+0.315)   Median rank=6,059,682 (Δ+20,359,371)
Without src_novelty        AUROC=0.364 (Δ-0.052)   Median rank=26,207,256 (Δ+211,796)
Without dst_novelty        AUROC=0.381 (Δ-0.035)   Median rank=26,181,064 (Δ+237,988)
Saved test_scored.pkl for bootstrap analysis

Done.
```

Notes on this stage:

- Baseline 2 encodes user, source host and destination host with
  LabelEncoders fitted on the union of both partitions, so that identifiers
  seen only in the test period do not raise an error. It is preserved
  unchanged as the pre-registered comparator.
- Baseline 2b one-hot encodes the same three fields with a vocabulary
  fitted on the training partition only (unseen identifiers become all-zero
  vectors). It is reported in the thesis as a diagnostic, not a comparator.
- The context-aware logistic regression is fitted on a partition of the
  training period at T_REF = 1002240: per-user history comes from the
  reference window before that second and the regression is fitted on the
  fit window after it, where the novelty flags vary. One MinMaxScaler is
  fitted on the fit-window features and applied unchanged to the test
  features.
- The printed ablation AUROC deltas are ablated minus full; Table 5.4 of the
  thesis reports full minus ablated, so the signs are reversed there. The
  printed rank deltas are already full minus ablated.

### Step 7 — Stage 5: bootstrap intervals, RQ1 tests, RQ2 intervals

```bash
python3 5_bootstrap.py
```

Expected output (05_bootstrap_output_batchC_rerun.txt, progress lines
omitted here):
```
=== BOOTSTRAP AUROC — 95% CONFIDENCE INTERVALS ===

Baseline 1 (Severity-Only)             AUROC = 0.492   [95% CI: 0.485 – 0.504]
Baseline 2 (Metadata-Only)             AUROC = 0.787   [95% CI: 0.762 – 0.807]
Baseline 2b (Metadata One-Hot)         AUROC = 1.000   [95% CI: 1.000 – 1.000]
Context-Aware (Equal Weights)          AUROC = 0.416   [95% CI: 0.352 – 0.482]
Context-Aware (Logistic Regression)    AUROC = 0.675   [95% CI: 0.619 – 0.725]

=== RQ1 SIGNIFICANCE TEST ===
Context-Aware (Logistic Regression) vs Baseline 2 (Metadata-Only)

AUROC difference (Context-Aware − Baseline 2 (Metadata-Only)):
  Mean:    -0.111
  95% CI:  [-0.174, -0.055]

  --> CI excludes zero, negative: context-aware SIGNIFICANTLY underperforms this baseline.

=== RQ1 SIGNIFICANCE TEST ===
Context-Aware (Logistic Regression) vs Baseline 2b (Metadata One-Hot)

AUROC difference (Context-Aware − Baseline 2b (Metadata One-Hot)):
  Mean:    -0.325
  95% CI:  [-0.381, -0.275]

  --> CI excludes zero, negative: context-aware SIGNIFICANTLY underperforms this baseline.

=== RQ2 ABLATION — PAIRED BOOTSTRAP 95% CONFIDENCE INTERVALS ===
ΔAUROC = full equal-weight model − model without the feature
(positive Δ: the feature helps; CI excluding zero: the change is
statistically distinguishable)

Without historical_fail_propensity   ΔAUROC = +0.005   [95% CI: -0.004, +0.018]   CI includes zero
Without time_deviation               ΔAUROC = -0.309   [95% CI: -0.375, -0.240]   CI excludes zero
Without src_novelty                  ΔAUROC = +0.052   [95% CI: +0.014, +0.097]   CI excludes zero
Without dst_novelty                  ΔAUROC = +0.035   [95% CI: +0.002, +0.075]   CI excludes zero

Done.
```

The bootstrap AUROC values are means over 1,000 resamples drawn on a
fixed 200,000-event negative subsample, so they can differ in the third
decimal from the full-test point estimates of stage 4 (0.675 against
0.676 for the context-aware logistic regression). The printed verdict
lines state the direction of each paired difference; what the metadata
comparisons establish for RQ1 is adjudicated in Sections 5.6, 5.8 and 6.1
of the thesis.

### Step 8 — Diagnostic and sensitivity scripts (independent of one another)

```bash
python3 "6 baseline2 mechanism.py"     # Experiments A to C, Section 5.6
python3 7_b2b_mechanism.py             # Experiments D and E, Section 5.6
python3 8_sampling_step0.py            # sampling regimes, Section 5.7
python3 9_sensitivityC.py              # Design C sensitivity analysis, Section 5.7
python3 10_novelty_base_rates.py       # novelty base rates, Section 5.9.3.2
python3 11_threshold_metrics.py        # Recall@k, F1, FPR, Section 4.7 and Appendix B.2
python3 12_feature_auroc.py            # single-feature AUROC, Section 5.4 and Appendix A.11
python3 13_lr_ablation.py              # refit ablation of the context LR, Section 5.4 and Appendix A.12
```

Each script reads the pipeline outputs (train.pkl, test.pkl,
test_features.pkl or test_scored.pkl) and prints the values reported in
the sections named above; the expected output of each is its capture file.
Script 9 writes test_scored_c.pkl and overwrites no other file. Script 11
asserts that the Precision@k it reproduces from test_scored.pkl equals the
stage-4 capture before printing anything, and stops with an assertion error
otherwise.

### Step 9 — Figures

```bash
python3 mkfigs.py
```

Parses 05_bootstrap_output_batchC_rerun.txt and
04_evaluate_output_batchB_rerun.txt, asserts every parsed value against
the reported results, and writes fig_auroc.pdf (Figure 5.1) and
fig_ablation.pdf (Figure 5.2).

---

## INTERMEDIATE FILES PRODUCED

| File | Size | Produced by | Required by |
|---|---|---|---|
| auth_working.txt | ~3.1 GB | preprocessing.sh | scripts 1 and 2 |
| train.pkl | ~0.7 GB | 2_split.py | scripts 3, 4, 6, 7, 8, 9, 13, check_dupes.py |
| test.pkl | ~2.5 GB | 2_split.py | scripts 3, 6, 7, 8, 9, check_dupes.py |
| test_features.pkl | ~4.8 GB | 3_features.py | scripts 4, 10, 12, 13 |
| test_scored.pkl | ~6.4 GB | 4_evaluate.py | scripts 5, 9 and 11 |
| test_scored_c.pkl | ~1.3 GB | 9_sensitivityC.py | none (kept as a record) |

---

## RANDOM SEEDS AND DETERMINISM

Randomness enters the pipeline in three scripts, each seeded; every other
stage is deterministic. seeds.txt records every value.

- preprocessing.sh: no random draw; the benign sample is every 1,000th row.
- 2_split.py: deterministic threshold 1,252,799.
- 3_features.py: deterministic.
- 4_evaluate.py: no random_state on any LogisticRegression; all are
  LogisticRegression(max_iter=1000, class_weight='balanced') with the
  default lbfgs solver, which is deterministic for a fixed input matrix.
  T_REF = 1002240 is a constant.
- 5_bootstrap.py: the 200,000-event negative subsample is drawn once with
  numpy.random.default_rng(42); iteration i draws its attack resample with
  random_state=i and its negative resample with random_state=i+10000.
- 6 baseline2 mechanism.py: the twenty permutations of the identity codes
  come from numpy.random.default_rng(42) (SEED = 42, N_PERMUTATIONS = 20).
- 9_sensitivityC.py: the same negative subsample and per-iteration seeds as
  5_bootstrap.py (default_rng(42); random_state=i and i+10000).

---

## KNOWN PROPERTIES OF THE PIPELINE

**Precision@k is 0.000 for every comparator.** With 82 attacks in
39,119,508 test events (about 1 in 477,000), the top 20, 100 and 1,000
positions contain no attack for any comparator, so Recall@k and F1 at those
cuts are 0.000 as well (script 11). At each scorer's natural decision cut F1
rounds to 0.000 for every comparator too: the metadata regression at p >= 0.5
alerts 11,620,488 events to recover 70 of the 82 attacks. The only non-zero
Precision@k and F1 belong to the diagnostic Baseline 2b, whose mechanism
Section 5.6 of the thesis establishes. AUROC and median attack rank are the metrics of
record; Appendix B.2 records the substitution.

**The equal-weight scorer is below 0.5.** Two of the four features
(historical_fail_propensity and time_deviation) have lower means for the
red-team events than for normal events. At the rank level only
time_deviation is inverted (single-feature AUROC 0.319, script 12); it
spans the whole scaled range and dominates the sum, so a uniform positive
weight ranks attacks downward. historical_fail_propensity used alone ranks
attacks at 0.705 despite its lower mean, because 44.45 percent of normal
events carry a propensity of exactly zero against 21.95 percent of attacks.
Section 5.9 of the thesis explains this.

**Users without training history score as maximally familiar.** They
receive propensity 0, mean hour 12 and both novelty flags 0. Section 6.3 of
the thesis records this as a limitation of the feature definitions.

**auth_working.txt contains 50,280 duplicated lines (0.099 percent).**
Measured with `sort auth_working.txt | uniq -d | wc -l`. The sampling
overlap accounts for 49,980 of them; the remainder are events that are
identical in auth.txt itself. check_dupes.py shows that none of the 82 test
or 610 training attack rows is an exact duplicate of another. The file is
preserved as it was when the results were produced.

**Label encoders of Baseline 2 see both partitions.** The integer codes for
user, source host and destination host are fitted on the union of training
and test identifiers. No label information crosses the split; the vocabulary
does. Baseline 2b's one-hot vocabulary is training-only.

---

## AI TOOL USAGE

The scripts in this artefact were prepared with the assistance of a
generative AI tool in two phases. The five pipeline scripts were written
from the dataset description, feature definitions, split boundary, scoring
methods and evaluation protocol specified by the author in the research
proposal, and were later corrected under the author's review. The
diagnostic, sensitivity, threshold-metric, single-feature and refit-ablation
scripts were written from designs specified and approved by the author
during the correction phase. The author reviewed,
executed and verified every script. No result value reported in the thesis
was produced, estimated or adjusted by an AI tool. The use is declared on the
university's List of Generative AI Tool Usages at the end of the thesis. The
original development session is cited as reference [69]; the correction-phase
sessions cannot be shared by link and are therefore not listed as a
reference, and the scripts and figures prepared or corrected in that phase
are marked as generated using AI where they appear (Appendix A captions,
Figures 5.1 and 5.2, Appendix B.1).

---

## CITATION

**Thesis:**
M. S. Mir, "Context-Aware Risk Scoring for Prioritizing Authentication
Alerts in Security Operations Centers," M.Sc. thesis, SRH University of
Applied Sciences Heidelberg, Campus Leipzig, 2026.

**Dataset:**
A. D. Kent, "Comprehensive, multi-source cyber-security events data set,"
Los Alamos National Laboratory, 2015. doi: 10.17021/1179829

---

## CONTACT

Muhammad Saad Mir
saadmir4561@gmail.com
SRH University of Applied Sciences Heidelberg, Campus Leipzig
Matriculation: 100001795
