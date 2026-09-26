import pandas as pd
import numpy as np
from scipy.stats import circmean

print("Loading training and test data...")
train = pd.read_pickle('train.pkl')
test  = pd.read_pickle('test.pkl')

print(f"Train: {len(train):,}  Test: {len(test):,}")

# Add hour of day (0-23) from timestamp
train['hour'] = (train['time'] % 86400) // 3600
test['hour']  = (test['time']  % 86400) // 3600

print("Computing features from training period only...")

# FEATURE 1: Historical fail propensity per user
# (failures / total attempts over the whole training period)
user_total_events = train.groupby('src_user').size().rename('total_events')
user_fail_events  = (train[train['success'] == 'Fail']
                      .groupby('src_user').size().rename('fail_events'))

historical_fail_propensity = ((user_fail_events / user_total_events)
                              .rename('historical_fail_propensity').fillna(0))

# FEATURE 2: Circular mean login hour per user in training period.
# Clock hours are cyclical (23 and 1 are two hours apart), so the
# arithmetic mean is replaced with the circular mean on a 24-hour cycle.
mean_hours = (train.groupby('src_user')['hour']
              .agg(lambda h: circmean(h, high=24, low=0))
              .rename('mean_hour'))

# FEATURE 3: Known source computers per user in training period
known_src = (train.groupby('src_user')['src_computer']
             .apply(set).rename('known_src'))

# FEATURE 4: Known destination computers per user in training period
known_dst = (train.groupby('src_user')['dst_computer']
             .apply(set).rename('known_dst'))

# Apply features to test set
print("Applying features to test set...")
test = test.join(historical_fail_propensity, on='src_user')
test = test.join(mean_hours, on='src_user')
test = test.join(known_src,  on='src_user')
test = test.join(known_dst,  on='src_user')

# Fill missing values for users not seen in training
test['historical_fail_propensity'] = test['historical_fail_propensity'].fillna(0)
test['mean_hour']  = test['mean_hour'].fillna(12)

# Time deviation: circular distance between this login's hour and the
# user's circular mean hour on the 24-hour cycle (range 0 to 12)
hour_diff = (test['hour'] - test['mean_hour']).abs()
test['time_deviation'] = np.minimum(hour_diff, 24 - hour_diff)

# Source host novelty: 1 if this computer was never seen in training
test['src_novelty'] = test.apply(
    lambda r: 1 if isinstance(r['known_src'], set)
              and r['src_computer'] not in r['known_src'] else 0, axis=1)

# Destination host novelty: 1 if this computer was never seen in training
test['dst_novelty'] = test.apply(
    lambda r: 1 if isinstance(r['known_dst'], set)
              and r['dst_computer'] not in r['known_dst'] else 0, axis=1)

# Show feature stats for attacks vs normal
feature_cols = ['historical_fail_propensity', 'time_deviation',
                'src_novelty', 'dst_novelty']
print("\nFeature means — ATTACKS vs NORMAL:")
attacks = test[test['label'] == 1]
normal  = test[test['label'] == 0]
for f in feature_cols:
    print(f"  {f:<20} attacks={attacks[f].mean():.3f}  normal={normal[f].mean():.3f}")

# Feature scaling now happens in 4_evaluate.py, fitted on training only.

# Save
test.to_pickle('test_features.pkl')
print("\nSaved test_features.pkl")
print("Feature engineering complete.")