import pandas as pd

COLS = ['time','src_user','dst_user','src_computer',
        'dst_computer','auth_type','logon_type',
        'auth_orient','success']

print("Loading auth data...")
auth = pd.read_csv('auth_working.txt', header=None, names=COLS)

redteam = pd.read_csv('redteam.txt', header=None,
                      names=['time','user','src_computer','dst_computer'])

print(f"Auth events loaded:  {len(auth):,}")
print(f"Attack events total: {len(redteam):,}")
print()
print("First 3 rows:")
print(auth.head(3))
print()
print("Success values:", auth['success'].unique())
print("Time range:", auth['time'].min(), "to", auth['time'].max())