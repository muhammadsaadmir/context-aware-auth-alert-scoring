"""
8_sampling_step0.py — sampling-regime measurement (Section 5.7)

Measures the sampling-regime overrepresentation in the working data.

preprocessing.sh built auth_working.txt from two regimes:
  DENSE  — complete histories for every account whose identifier contains
           one of the 91 hand-listed red-team substrings (substring grep,
           so a superset of the red-team accounts);
  SPARSE — a 1-in-1000 systematic sample of everything else.

This script classifies each src_user by the same 91 substrings and reports,
for both regimes: user counts, row counts, events per user (mean and
median), the resulting overrepresentation ratio, and the regime shares of
all rows and of the test-period normal rows. It also reports events per
user for the 98 users named in redteam.txt.

Measured from train.pkl + test.pkl, which together contain all 51,031,056
rows of auth_working.txt. Read-only: writes nothing.

CAVEAT (printed in the output as well): the original grep matched the
substrings anywhere in the raw line, so a row can be in the dense regime
through a field other than src_user. Classification here is by src_user
only; dense-regime rows are therefore counted conservatively.
"""

import pandas as pd
import numpy as np
import re

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
assert len(PATTERNS) == 91, "pattern list must match preprocessing.sh exactly"

print("Loading data ...")
train = pd.read_pickle('train.pkl')
test = pd.read_pickle('test.pkl')
n_train, n_test = len(train), len(test)
print(f"Train {n_train:,} rows | Test {n_test:,} rows | combined {n_train + n_test:,} "
      f"(should equal 51,031,056)")

counts = pd.concat([train['src_user'], test['src_user']],
                   ignore_index=True).value_counts()
uniq = counts.index.to_series().astype(str)

regex = re.compile("|".join(PATTERNS))
dense_users = set(uniq[uniq.str.contains(regex, regex=True, na=False)])
print(f"\nDistinct src_user values: {len(counts):,}")
print(f"Pattern-matched (DENSE regime) users: {len(dense_users):,}")
print(f"Remaining (SPARSE regime) users:      {len(counts) - len(dense_users):,}")

dense_counts = counts[counts.index.isin(dense_users)]
sparse_counts = counts[~counts.index.isin(dense_users)]

def describe(name, c):
    print(f"\n{name}: users={len(c):,}  rows={int(c.sum()):,}  "
          f"mean events/user={c.mean():,.1f}  median={c.median():,.0f}")
    return c.mean(), c.median()

d_mean, d_med = describe("DENSE regime", dense_counts)
s_mean, s_med = describe("SPARSE regime", sparse_counts)

print(f"\nOverrepresentation ratio (dense/sparse, mean events per user):   "
      f"{d_mean / s_mean:,.1f}x")
print(f"Overrepresentation ratio (dense/sparse, median events per user): "
      f"{d_med / s_med:,.1f}x")
total_rows = counts.sum()
print(f"\nShare of ALL rows in the dense regime: "
      f"{int(dense_counts.sum()):,}/{int(total_rows):,} = "
      f"{100 * dense_counts.sum() / total_rows:.2f}%")

test_normals = test[test['label'] == 0]
tn_dense = test_normals['src_user'].isin(dense_users)
print(f"Share of TEST NORMAL rows in the dense regime: "
      f"{int(tn_dense.sum()):,}/{len(test_normals):,} = {100 * tn_dense.mean():.2f}%")

redteam = pd.read_csv('redteam.txt', header=None,
                      names=['time', 'user', 'src_computer', 'dst_computer'])
rt_users = set(redteam['user'].unique())
rt_counts = counts[counts.index.isin(rt_users)]
print(f"\nRed-team users (from redteam.txt): {len(rt_users)} distinct; "
      f"{len(rt_counts)} present as src_user in the working data")
print(f"  rows={int(rt_counts.sum()):,}  mean events/user={rt_counts.mean():,.1f}  "
      f"median={rt_counts.median():,.0f}")
print(f"  red-team mean vs sparse-regime mean: {rt_counts.mean() / s_mean:,.1f}x")

print("\nCaveat: the original grep matched substrings anywhere in the raw line;")
print("classification here is by src_user only, so dense-regime rows are")
print("counted conservatively.")

print("\nDone.")
