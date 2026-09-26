import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import circmean
from scipy.sparse import hstack, csr_matrix

print("Loading data...")
train = pd.read_pickle('train.pkl')
test  = pd.read_pickle('test_features.pkl')

y_test = test['label']
print(f"Test events: {len(test):,}  Attacks: {y_test.sum()}")

train['hour'] = (train['time'] % 86400) // 3600

redteam = pd.read_csv('redteam.txt', header=None,
                       names=['time','user','src_computer','dst_computer'])
attack_keys = set(zip(redteam['time'], redteam['src_computer'], redteam['dst_computer']))
train['label'] = train.apply(
    lambda r: 1 if (r['time'], r['src_computer'], r['dst_computer']) in attack_keys else 0, axis=1)
y_train = train['label']

# ── BASELINE 1: Severity-Only ──────────────────────────────────────────────
test['score_b1'] = (test['success'] == 'Fail').astype(float)

# ── BASELINE 2: Metadata-Only Logistic Regression ──────────────────────────
# Preserved unchanged as the documented historical comparator (integer
# encoding via LabelEncoder). The corrected encoding is Baseline 2b below.
le_user = LabelEncoder(); le_src = LabelEncoder(); le_dst = LabelEncoder()
all_users = pd.concat([train['src_user'], test['src_user']]).fillna('UNK')
all_src   = pd.concat([train['src_computer'], test['src_computer']]).fillna('UNK')
all_dst   = pd.concat([train['dst_computer'], test['dst_computer']]).fillna('UNK')
le_user.fit(all_users); le_src.fit(all_src); le_dst.fit(all_dst)

def build_meta(df):
    return pd.DataFrame({
        'user': le_user.transform(df['src_user'].fillna('UNK')),
        'src':  le_src.transform(df['src_computer'].fillna('UNK')),
        'dst':  le_dst.transform(df['dst_computer'].fillna('UNK')),
        'hour': df['hour'],
        'fail': (df['success'] == 'Fail').astype(int)
    })

print("Training metadata-only baseline...")
lr_meta = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_meta.fit(build_meta(train), y_train)
test['score_b2'] = lr_meta.predict_proba(build_meta(test))[:, 1]

# ── BASELINE 2b: Metadata-Only LR, one-hot encoding (corrected) ────────────
# Nominal identifiers are one-hot encoded instead of being treated as
# ordered integers. Vocabulary is fitted on the TRAINING partition only;
# identifiers unseen in training become all-zero vectors
# (handle_unknown='ignore').
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(train[['src_user', 'src_computer', 'dst_computer']].fillna('UNK'))

def build_meta_onehot(df):
    cats  = ohe.transform(df[['src_user', 'src_computer', 'dst_computer']].fillna('UNK'))
    dense = np.column_stack([df['hour'].to_numpy(),
                             (df['success'] == 'Fail').astype(int).to_numpy()])
    return hstack([cats, csr_matrix(dense)], format='csr')

print("Training metadata-only baseline (one-hot)...")
lr_meta_oh = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_meta_oh.fit(build_meta_onehot(train), y_train)
test['score_b2_onehot'] = lr_meta_oh.predict_proba(build_meta_onehot(test))[:, 1]

# ── CONTEXT FEATURES — three-way partition inside the training period ──────
# Pre-registered boundary: T_REF = 1002240 (80/20 split of training time).
# The reference window (time < T_REF) supplies ALL per-user history; the fit
# window (time >= T_REF) is where training-side features are computed and the
# context LR is trained, so the novelty flags vary there instead of being
# structurally zero. This mirrors the train/test relationship: features
# always come from history before the scoring window. Test features stay as
# produced by 3_features.py (full-training history).
print("Building context features for training set (three-way partition)...")
T_REF = 1002240
reference = train[train['time'] < T_REF]
fit = train[train['time'] >= T_REF].copy()
print(f"Reference window: {len(reference):,} rows   Fit window: {len(fit):,} rows   "
      f"Attacks in fit window: {fit['label'].sum()}")

user_total = reference.groupby('src_user').size().rename('total_events')
user_fail  = reference[reference['success']=='Fail'].groupby('src_user').size().rename('fail_events')
hist_fail_lookup = (user_fail / user_total).rename('historical_fail_propensity').fillna(0)
mean_hour_lookup = (reference.groupby('src_user')['hour']
                    .agg(lambda h: circmean(h, high=24, low=0)).rename('mean_hour'))
known_src_lookup = reference.groupby('src_user')['src_computer'].apply(set).rename('known_src')
known_dst_lookup = reference.groupby('src_user')['dst_computer'].apply(set).rename('known_dst')

fit = fit.join(hist_fail_lookup, on='src_user')
fit = fit.join(mean_hour_lookup, on='src_user')
fit = fit.join(known_src_lookup, on='src_user')
fit = fit.join(known_dst_lookup, on='src_user')
fit['historical_fail_propensity'] = fit['historical_fail_propensity'].fillna(0)
fit['mean_hour'] = fit['mean_hour'].fillna(12)
hour_diff = (fit['hour'] - fit['mean_hour']).abs()
fit['time_deviation'] = np.minimum(hour_diff, 24 - hour_diff)
fit['src_novelty'] = fit.apply(
    lambda r: 1 if isinstance(r['known_src'], set) and r['src_computer'] not in r['known_src'] else 0, axis=1)
fit['dst_novelty'] = fit.apply(
    lambda r: 1 if isinstance(r['known_dst'], set) and r['dst_computer'] not in r['known_dst'] else 0, axis=1)

context_cols = ['historical_fail_propensity', 'time_deviation', 'src_novelty', 'dst_novelty']
# Corrected scaling: one scaler, fitted once on the fit-window training
# features, applied once to the unscaled test features from 3_features.py.
scaler_ctx = MinMaxScaler()
fit[context_cols]  = scaler_ctx.fit_transform(fit[context_cols].fillna(0))
test[context_cols] = scaler_ctx.transform(test[context_cols].fillna(0))

# ── CONTEXT-AWARE SCORER — Option A: Equal weights (original pre-registered method) ──
w_fail, w_time, w_src, w_dst = 0.25, 0.25, 0.25, 0.25
test['score_ctx_equal'] = (
    w_fail * test['historical_fail_propensity'] + w_time * test['time_deviation'] +
    w_src  * test['src_novelty'] + w_dst * test['dst_novelty']
)

# ── CONTEXT-AWARE SCORER — Option B: Logistic regression (weights learned from train) ──
print("Training context-aware logistic regression (4 context features only)...")
lr_ctx = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_ctx.fit(fit[context_cols], fit['label'])
test['score_ctx_lr'] = lr_ctx.predict_proba(test[context_cols])[:, 1]

print("Learned context feature weights (logistic regression coefficients):")
for name, coef in zip(context_cols, lr_ctx.coef_[0]):
    print(f"  {name:<18} {coef:+.3f}")

# ── EVALUATION ───────────────────────────────────────────────────────────
def precision_at_k(scores, labels, k):
    top_k_idx = scores.nlargest(k).index
    return labels.loc[top_k_idx].sum() / k

def median_attack_rank(scores, labels):
    ranks = scores.rank(ascending=False)
    return ranks[labels == 1].median(), len(scores)

def full_report(name, score_col):
    scores = test[score_col]
    auroc  = roc_auc_score(y_test, scores)
    p20    = precision_at_k(scores, y_test, 20)
    p100   = precision_at_k(scores, y_test, 100)
    p1000  = precision_at_k(scores, y_test, 1000)
    med_rank, total = median_attack_rank(scores, y_test)
    print(f"\n{name}")
    print(f"  AUROC:              {auroc:.3f}")
    print(f"  Precision@20:       {p20:.3f}")
    print(f"  Precision@100:      {p100:.3f}")
    print(f"  Precision@1000:     {p1000:.3f}")
    print(f"  Median attack rank: {med_rank:,.0f} out of {total:,}")
    print(f"  (random expectation = {total/2:,.0f}; lower is better)")

print("\n=== RESULTS ===")
full_report("Baseline 1 (Severity-Only)",         "score_b1")
full_report("Baseline 2 (Metadata-Only)",         "score_b2")
full_report("Baseline 2b (Metadata One-Hot)",     "score_b2_onehot")
full_report("Context-Aware (Equal Weights)",      "score_ctx_equal")
full_report("Context-Aware (Logistic Regression)","score_ctx_lr")

# ── ABLATION (RQ2) — on the equal-weight score, as originally specified ────
print("\n=== ABLATION STUDY (equal-weight context score) ===")
features = {'historical_fail_propensity': w_fail, 'time_deviation': w_time,
            'src_novelty': w_src, 'dst_novelty': w_dst}
base_auroc = roc_auc_score(y_test, test['score_ctx_equal'])
base_med_rank, total = median_attack_rank(test['score_ctx_equal'], y_test)
print(f"Full model — AUROC={base_auroc:.3f}  Median rank={base_med_rank:,.0f} out of {total:,}\n")
for drop_feat in features:
    remaining = {k: v for k, v in features.items() if k != drop_feat}
    score = sum(test[k] * v for k, v in remaining.items())
    auroc = roc_auc_score(y_test, score)
    med_rank, _ = median_attack_rank(score, y_test)
    print(f"Without {drop_feat:<18} AUROC={auroc:.3f} (Δ{auroc-base_auroc:+.3f})   "
          f"Median rank={med_rank:,.0f} (Δ{base_med_rank-med_rank:+,.0f})")
test.to_pickle('test_scored.pkl')
print("Saved test_scored.pkl for bootstrap analysis")
print("\nDone.")
