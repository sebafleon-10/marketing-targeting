"""Stumptown wholesale sampling (teaching case, simulated data).

15,000 accounts got a free 1-lb bag; buyer = reordered 20 lbs. With only
15K rows every score is out-of-fold from 5-fold cross-validation, so no
account is scored by a model that saw it.
"""
import numpy as np, pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from common import RAW, SEED, gains, save

OFFER = 13 + 7                 # shipping + cost of goods per sample
WIN = 440 - 30 - 140           # reorder revenue minus shipping and goods
REMAINING = 45_000
breakeven = OFFER / WIN
df = pd.read_csv(RAW / "StumptownWholesale.csv")
X_COLS = [c for c in df.columns if c not in ("customer_id", "buyer")]
y = df.buyer.to_numpy()
print(f"conversion {y.mean():.4f}  breakeven {breakeven:.4f}")

oof = {"Logistic regression": np.zeros(len(df)), "LightGBM": np.zeros(len(df))}
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=60, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, seed=SEED)
rounds = []
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, y)):
    tr, te = df.iloc[a], df.iloc[b]
    m = sm.Logit(tr.buyer, sm.add_constant(tr[X_COLS])).fit(disp=0)
    oof["Logistic regression"][b] = m.predict(sm.add_constant(te[X_COLS], has_constant="add"))
    fit, val = train_test_split(tr, test_size=0.2, stratify=tr.buyer, random_state=SEED + k)
    es = lgb.train(params, lgb.Dataset(fit[X_COLS], fit.buyer), 3000, valid_sets=[lgb.Dataset(val[X_COLS], val.buyer)],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds.append(es.best_iteration)
    g = lgb.train(params, lgb.Dataset(tr[X_COLS], tr.buyer), es.best_iteration)
    oof["LightGBM"][b] = g.predict(te[X_COLS])

def score(mask, label):
    n = int(mask.sum()); wins = int(y[mask].sum())
    profit = wins * WIN - n * OFFER
    scale = REMAINING / len(df)
    r = {"policy": label, "targeted_share": n / len(df), "targeted": round(n * scale), "response_rate": wins / n if n else 0.0,
         "profit": profit * scale, "roi": profit / (n * OFFER) if n else 0.0}
    print(f"{label:22s} target {r['targeted_share']:.3f}  conv {r['response_rate']:.4f}  profit ${r['profit']:,.0f}  ROI {r['roi']:.1%}")
    return r

policies = [score(np.ones(len(df), bool), "Sample everyone")]
for k, p in oof.items():
    policies.append(score(p > breakeven, k))
auc = {k: roc_auc_score(y, p) for k, p in oof.items()}
print({k: round(v, 4) for k, v in auc.items()})

# Confounding check from the case: seasonal single-origin alone vs in the full model.
full = sm.Logit(df.buyer, sm.add_constant(df[X_COLS])).fit(disp=0)
alone = sm.Logit(df.buyer, sm.add_constant(df[["seasonal_single_origin_cases"]])).fit(disp=0)
coef = {"full": full.params["seasonal_single_origin_cases"], "alone": alone.params["seasonal_single_origin_cases"]}
print("seasonal coef full vs alone:", {k: round(v, 4) for k, v in coef.items()})

deciles = []
for k, p in oof.items():
    d = pd.qcut(-p, 10, labels=False) + 1
    deciles.append({"model": k, "rates": pd.Series(y).groupby(d).mean().round(4).tolist()})
save("stumptown", {"conversion": y.mean(), "breakeven": breakeven, "offerCost": OFFER, "winValue": WIN, "remaining": REMAINING,
                   "policies": policies, "auc": auc, "rounds": rounds, "seasonalCoef": coef, "deciles": deciles,
                   "gains": {k: gains(p, y) for k, p in oof.items()}})
