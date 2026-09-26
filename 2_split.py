import pandas as pd

COLS = ['time','src_user','dst_user','src_computer',
        'dst_computer','auth_type','logon_type',
        'auth_orient','success']

print("Loading data...")
auth = pd.read_csv('auth_working.txt', header=None, names=COLS)
redteam = pd.read_csv('redteam.txt', header=None,
                      names=['time','user','src_computer','dst_computer'])

# Temporal split at 25% of the observed time window: earlier quarter trains, later
# three quarters is the evaluation set. Set to place enough red-team events in the
# test period, since the labels are concentrated in the later part of the window.
split_time = int(auth['time'].max() * 0.25)
print(f"Split time: {split_time}")

train = auth[auth['time'] <= split_time].copy()
test  = auth[auth['time'] >  split_time].copy()

print(f"Training events: {len(train):,}")
print(f"Test events:     {len(test):,}")

# Label test events: 1 = attack, 0 = normal
attack_keys = set(zip(redteam['time'],
                      redteam['src_computer'],
                      redteam['dst_computer']))

test['label'] = test.apply(
    lambda r: 1 if (r['time'], r['src_computer'], r['dst_computer'])
              in attack_keys else 0, axis=1)

print(f"\nAttacks found in test set: {test['label'].sum()}")
print(f"Normal events in test set: {(test['label']==0).sum():,}")

# Save for next scripts
train.to_pickle('train.pkl')
test.to_pickle('test.pkl')
print("\nSaved train.pkl and test.pkl")