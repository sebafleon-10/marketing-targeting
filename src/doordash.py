"""DoorDash push-notification targeting (teaching case, simulated data).

Every one of the 70,000 test customers received the push, so this measures
who responds, not who responds *because* of the push (no control group).
Targets are chosen on one half of the data and scored on the other half.
"""
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import RAW, SEED, gains, save

COST = 0.96          # case: 3% opt-out chance x $32 LTV gap
BASE = 36_930_000    # customers not in the test
df = pd.read_csv(RAW / "DoorDash.csv")
df["ios"] = (df.device == "iOS").astype(int)
X_COLS = ["recency", "frequency", "monetary", "ios", "age"]
tr, te = train_test_split(df, test_size=0.5, stratify=df.response, random_state=SEED)

avg_profit = tr.loc[tr.response == 1, "profit"].mean()
breakeven = COST / avg_profit
print(f"response {df.response.mean():.4f}  profit per responder ${avg_profit:.2f}  breakeven {breakeven:.4f}")

def score(mask, label):
    n = int(mask.sum())
    profit = te.loc[mask, "profit"].sum() - COST * n
    scale = BASE / len(te)
    r = {"policy": label, "targeted_share": n / len(te), "targeted": round(n * scale),
         "response_rate": float(te.loc[mask, "response"].mean()) if n else 0.0,
         "profit": profit * scale, "roi": profit / (COST * n) if n else 0.0}
    print(f"{label:22s} target {r['targeted_share']:.3f}  resp {r['response_rate']:.4f}  profit ${r['profit']/1e6:,.2f}M  ROI {r['roi']:.2%}")
    return r

policies = [score(np.ones(len(te), bool), "Send to everyone")]

# RFM: quintiles from the training half's distribution (1 = best), cells above breakeven get the push.
def quint(train_col, col):
    ref = np.sort(train_col.to_numpy())
    ecdf = np.searchsorted(ref, col.to_numpy(), side="right") / len(ref)
    return np.clip(np.ceil(ecdf * 5), 1, 5).astype(int)
def rfm(frame):
    r = quint(tr.recency, frame.recency)            # fewer days since last order is better
    f = 6 - quint(tr.frequency, frame.frequency)    # more orders is better
    m = 6 - quint(tr.monetary, frame.monetary)      # more spend is better
    return pd.Series(r * 100 + f * 10 + m, index=frame.index)
tr_cell, te_cell = rfm(tr), rfm(te)
cell_rate = tr.groupby(tr_cell).response.mean()
policies.append(score(te_cell.map(cell_rate).fillna(0).to_numpy() > breakeven, "RFM cells"))
rfm_score = te_cell.map(cell_rate).fillna(0).to_numpy()

logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[X_COLS], tr.response)
p_log = logit.predict_proba(te[X_COLS])[:, 1]
policies.append(score(p_log > breakeven, "Logistic regression"))

# Gradient boosting with early stopping on a slice of the training half.
fit, val = train_test_split(tr, test_size=0.25, stratify=tr.response, random_state=SEED)
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.9, verbose=-1, seed=SEED)
b = lgb.train(params, lgb.Dataset(fit[X_COLS], fit.response), 2000, valid_sets=[lgb.Dataset(val[X_COLS], val.response)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
gbm = lgb.train(params, lgb.Dataset(tr[X_COLS], tr.response), b.best_iteration)
p_gbm = gbm.predict(te[X_COLS])
policies.append(score(p_gbm > breakeven, "LightGBM"))

yt = te.response.to_numpy()
auc = {"RFM cells": roc_auc_score(yt, rfm_score), "Logistic regression": roc_auc_score(yt, p_log), "LightGBM": roc_auc_score(yt, p_gbm)}
print({k: round(v, 4) for k, v in auc.items()})
save("doordash", {"responseRate": df.response.mean(), "profitPerResponder": avg_profit, "breakeven": breakeven, "cost": COST,
                  "base": BASE, "policies": policies, "auc": auc, "rounds": b.best_iteration,
                  "gains": {"RFM cells": gains(rfm_score, yt), "Logistic regression": gains(p_log, yt), "LightGBM": gains(p_gbm, yt)}})

# Robustness: repeat the whole split-target-score loop over 20 random halves.
rep = {k: [] for k in ["Send to everyone", "RFM cells", "Logistic regression", "LightGBM"]}
for s in range(20):
    a, b2 = train_test_split(df, test_size=0.5, stratify=df.response, random_state=1000 + s)
    be = COST / a.loc[a.response == 1, "profit"].mean()
    scale = BASE / len(b2)
    val_profit = lambda m: (b2.loc[m, "profit"].sum() - COST * m.sum()) * scale
    tr, te = a, b2  # rfm() reads tr for its quintile edges
    rate = a.groupby(rfm(a)).response.mean()
    lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(a[X_COLS], a.response)
    gb = lgb.train(params, lgb.Dataset(a[X_COLS], a.response), b.best_iteration)
    rep["Send to everyone"].append(val_profit(np.ones(len(b2), bool)))
    rep["RFM cells"].append(val_profit(rfm(b2).map(rate).fillna(0).to_numpy() > be))
    rep["Logistic regression"].append(val_profit(lg.predict_proba(b2[X_COLS])[:, 1] > be))
    rep["LightGBM"].append(val_profit(gb.predict(b2[X_COLS]) > be))
robust = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in rep.items()}
gap = np.array(rep["Logistic regression"]) - np.array(rep["RFM cells"])
robust["logistic_minus_rfm"] = {"mean": float(gap.mean()), "sd": float(gap.std()), "share_positive": float((gap > 0).mean())}
for k, v in robust.items(): print("robust", k, {kk: round(vv / 1e6, 3) if kk != "share_positive" else vv for kk, vv in v.items()})
import json
res = json.load(open("results/doordash.json")); res["robust"] = robust
json.dump(res, open("results/doordash.json", "w"), indent=2, default=float)
