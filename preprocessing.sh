#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# preprocessing.sh
# Thesis: Context-Aware Risk Scoring for Prioritizing
#         Authentication Alerts in Security Operations Centers
# Author: Muhammad Saad Mir | Matriculation: 100001795
# SRH University of Applied Sciences Heidelberg, Campus Leipzig
# Supervisor: Prof. Dr. Klaus Dieter Schwarz
#
# PURPOSE:
#   Reproduces auth_working.txt from the raw LANL dataset files.
#   Run this script ONLY if auth_working.txt is missing.
#   If auth_working.txt already exists, skip this script entirely
#   and proceed directly to the Python scripts.
#
# INPUT FILES REQUIRED (download from https://csr.lanl.gov/data/cyber1/):
#   auth.txt      — full authentication log (~73.4 GB, about 1.05 billion rows)
#   redteam.txt   — red-team attack labels (749 events)
#
# OUTPUT:
#   auth_working.txt — 51,031,056 rows, ~3.1 GB, sorted by timestamp
#
# RUNTIME: approximately 30-60 minutes for Steps 1 and 2, plus 5-15 minutes
#          for the Step 3 sort, on a standard laptop.
#
# DISK: Step 3 sorts a 3.1 GB file and needs comparable free temporary space
#       in addition to the output. Ensure roughly 10 GB free beyond auth.txt.
# ─────────────────────────────────────────────────────────────────────────────

set -e  # Stop immediately if any command fails

echo "========================================================"
echo " LANL Authentication Log Preprocessing"
echo " Thesis reproducibility artefact"
echo "========================================================"
echo ""

# ── Verify input files exist ──────────────────────────────────────────────────
if [ ! -f "auth.txt" ]; then
    echo "ERROR: auth.txt not found in current directory."
    echo "Download from: https://csr.lanl.gov/data/cyber1/"
    echo "Then decompress: gunzip auth.txt.gz"
    exit 1
fi

if [ ! -f "redteam.txt" ]; then
    echo "ERROR: redteam.txt not found in current directory."
    echo "Download from: https://csr.lanl.gov/data/cyber1/"
    echo "Then decompress: gunzip redteam.txt.gz"
    exit 1
fi

echo "Input files verified."
echo "auth.txt size:    $(du -sh auth.txt | cut -f1)"
echo "redteam.txt size: $(du -sh redteam.txt | cut -f1)"
echo ""

# ── Step 1: Extract all events involving red-team users ───────────────────────
# These are the users who appear in redteam.txt as attackers.
# We keep ALL their authentication events (both normal and attack-period)
# so that training-period behaviour baselines can be computed for them.
#
# TWO PROPERTIES OF THIS EXTRACTION ARE DOCUMENTED HONESTLY:
#
# (1) The grep pattern is a substring match over the whole line, not a
#     field-anchored match on an account identifier. "U12" therefore also
#     matches U120, U124, U1289 and so on. The extraction is a superset of
#     the red-team accounts, which is harmless for the study: extra benign
#     events only enlarge the normal population.
#
# (2) The account list below was derived by hand from redteam.txt and is
#     not complete. Section 4.4 of the thesis reports the consequence: of
#     the 633 distinct training-period red-team keys, 609 are present in
#     auth_working.txt and 24 are absent. All 80 test-period keys are
#     present, so the evaluation set is complete and no reported result is
#     affected. A more robust reconstruction would derive the list at
#     runtime with:
#         cut -d',' -f2 redteam.txt | cut -d'@' -f1 | sort -u
#     This script preserves the hand-derived list because it is the list
#     that produced the reported results.
#
# Red-team users identified from redteam.txt:
# U748, U66, U737, U293, U1723, U3635, U162, U3005, U8946, U8601,
# U218, U342, U4448, U636, U825, U1653, U4978, U5087, U9947, U9763,
# U4353, U1450, U374, U2575, U882, U8777, U3718, U314, U642, U6572,
# U2837, U349, U1600, U250, U4856, U9407, U4112, U7375, U7507, U415,
# U1145, U1480, U453, U207, U1289, U1519, U3486, U1592, U1025, U9263,
# U655, U86, U3549, U8170, U679, U7311, U524, U1133, U78, U3764,
# U212, U995, U795, U6691, U2231, U7594, U114, U1106, U3575, U3206,
# U227, U1306, U8840, U1467, U3406, U10379, U8168, U3277, U7761,
# U7004, U7394, U1048, U5254, U6764, U1569, U1581, U1789, U13,
# U12, U20, U24

echo "Step 1: Extracting red-team user events from auth.txt..."
echo "        (This will take 20-40 minutes)"

grep -E "U748|U66|U737|U293|U1723|U3635|U162|U3005|U8946|U8601|\
U218|U342|U4448|U636|U825|U1653|U4978|U5087|U9947|U9763|\
U4353|U1450|U374|U2575|U882|U8777|U3718|U314|U642|U6572|\
U2837|U349|U1600|U250|U4856|U9407|U4112|U7375|U7507|U415|\
U1145|U1480|U453|U207|U1289|U1519|U3486|U1592|U1025|U9263|\
U655|U86|U3549|U8170|U679|U7311|U524|U1133|U78|U3764|\
U212|U995|U795|U6691|U2231|U7594|U114|U1106|U3575|U3206|\
U227|U1306|U8840|U1467|U3406|U10379|U8168|U3277|U7761|\
U7004|U7394|U1048|U5254|U6764|U1569|U1581|U1789|U13|\
U12|U20|U24" auth.txt > auth_attacks.txt

ATTACK_LINES=$(wc -l < auth_attacks.txt)
echo "        Done. Lines extracted: $ATTACK_LINES"
echo "        Expected: 49,979,626"
echo ""

# ── Step 2: Sample 1-in-1000 normal events ────────────────────────────────────
# A systematic sample of every 1,000th row from the full auth.txt provides
# a representative baseline of normal authentication behaviour.
# This is standard practice for large-scale imbalanced security datasets.
# Sampling rate: 1 in 1,000 rows

echo "Step 2: Sampling 1-in-1000 normal events..."
echo "        (This will take 10-20 minutes)"

awk 'NR % 1000 == 0' auth.txt > auth_normal_sample.txt

NORMAL_LINES=$(wc -l < auth_normal_sample.txt)
echo "        Done. Lines sampled: $NORMAL_LINES"
echo "        Expected: 1,051,430 (deterministic, not approximate)"
echo ""

# ── Step 3: Combine and sort by timestamp ────────────────────────────────────
# The temporal split in 2_split.py requires events to be sorted by time
# (column 1). Combine both files and sort numerically on column 1.
#
# NOTE ON OVERLAP: Step 2 samples every 1000th row of the FULL auth.txt,
# including rows Step 1 already extracted. The two files are concatenated
# without de-duplication, which is why 49,979,626 + 1,051,430 sums exactly
# to 51,031,056. Measured with
#     sort auth_working.txt | uniq -d | wc -l
# the working file contains 50,280 distinct lines that occur more than
# once, about 0.099% of the file. The predicted count from the sampling
# overlap alone is 49,980 (one in every 1,000 of the 49,979,626 rows from
# Step 1); the remaining ~300 are events that are genuinely identical in
# auth.txt, that is, the same account authenticating between the same
# host pair within the same second. No labelled event is affected: an exact
# nine-field comparison across both partitions returns zero duplicate attack
# rows, so the 82 test-period and 610 training-period labelled rows are all
# distinct events (see check_dupes.py). The duplication is confined to the
# benign population and is preserved here because it is the state in which
# the reported results were produced.

echo "Step 3: Combining files and sorting by timestamp..."
echo "        (This will take 5-15 minutes and needs ~10 GB temporary space)"

cat auth_attacks.txt auth_normal_sample.txt | \
    sort -t',' -k1,1n > auth_working.txt

TOTAL_LINES=$(wc -l < auth_working.txt)
TOTAL_SIZE=$(du -sh auth_working.txt | cut -f1)

echo "        Done."
echo ""

# ── Verification ──────────────────────────────────────────────────────────────
echo "========================================================"
echo " PREPROCESSING COMPLETE"
echo "========================================================"
echo ""
echo " Output file:  auth_working.txt"
echo " Total rows:   $TOTAL_LINES"
echo " File size:    $TOTAL_SIZE"
echo ""
echo " Expected:     51,031,056 rows | 3.1 GB"
echo ""

if [ "$TOTAL_LINES" -eq 51031056 ]; then
    echo " Verification: PASSED — row count matches expected value"
else
    echo " Verification: WARNING — row count differs from expected"
    echo "               Expected 51,031,056, got $TOTAL_LINES"
    echo "               Results may differ slightly from thesis values"
fi

echo ""
echo " Next steps:"
echo "   python3 1_load_check.py"
echo "   python3 2_split.py"
echo "   python3 3_features.py"
echo "   python3 4_evaluate.py"
echo "   python3 5_bootstrap.py"
echo ""
echo " Intermediate files (can be deleted after auth_working.txt is created):"
echo "   rm auth_attacks.txt auth_normal_sample.txt"
echo "========================================================"