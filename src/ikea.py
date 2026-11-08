"""IKEA next-product-to-buy (teaching case, simulated data).

Each customer's most recent email came from a department picked at random
(1 in 7), so any "which email should this customer get" policy can be scored
honestly on the validation split with inverse propensity weighting: keep
the customers whose random email happens to match the policy, weight by 7.
Buyers were oversampled to 50 percent; weights restore the true 0.84 percent.
"""
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import RAW, SEED, save

O, R = 0.0084, 0.50
MARGIN = 0.35
CAMPAIGN = 10_000_000
DEPTS = ["LivingRoom", "Bedroom", "KitchenDining", "Storage", "Lighting", "HomeDecor", "Outdoor"]
K = len(DEPTS)

df = pd.read_csv(RAW / "IKEA.csv")
df = pd.get_dummies(df, columns=["age"], drop_first=True, dtype=float)
df["w"] = np.where(df.buy == 1, O / R, (1 - O) / (1 - R))
FEATS = [c for c in df.columns if c not in ("customer_id", "buy", "order_size", "message", "training", "w")]
tr, va = df[df.training == 1].copy(), df[df.training == 0].copy()
profit_per_buy = (tr[tr.buy == 1].groupby("message").order_size.mean() * MARGIN).reindex(DEPTS)
print("profit per purchase:", profit_per_buy.round(2).to_dict())

# Model A, the class approach: one weighted logistic regression per department.
sc = StandardScaler().fit(tr[FEATS])
pA = np.zeros((len(va), K))
for j, d in enumerate(DEPTS):
    t = tr[tr.message == d]
    m = LogisticRegression(max_iter=5000).fit(sc.transform(t[FEATS]), t.buy, sample_weight=t.w)
    pA[:, j] = m.predict_proba(sc.transform(va[FEATS]))[:, 1]

# Model B: one gradient-boosted model with the department as an input, so it can learn
# which customers respond to which message, then ask it about all seven messages.
X = lambda frame, msg: frame[FEATS].assign(msg=pd.Categorical(msg, categories=DEPTS))
fit = tr.sample(frac=0.8, random_state=SEED); hold = tr.drop(fit.index)
params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, seed=SEED)
es = lgb.train(params, lgb.Dataset(X(fit, fit.message), fit.buy, weight=fit.w), 3000,
               valid_sets=[lgb.Dataset(X(hold, hold.message), hold.buy, weight=hold.w)],
               callbacks=[lgb.early_stopping(100, verbose=False)])
gbm = lgb.train(params, lgb.Dataset(X(tr, tr.message), tr.buy, weight=tr.w), es.best_iteration)
pB = np.column_stack([gbm.predict(X(va, [d] * len(va))) for d in DEPTS])

va_profit = (va.buy * va.order_size * MARGIN).to_numpy()
w = va.w.to_numpy(); msg = va.message.map({d: j for j, d in enumerate(DEPTS)}).to_numpy()

def ips(choice, idx=None):
    """Average profit per customer if each customer got email `choice` (department index)."""
    i = np.arange(len(va)) if idx is None else idx
    hit = (msg[i] == choice[i]).astype(float)
    return float(np.sum(w[i] * hit * K * va_profit[i]) / np.sum(w[i]))

rng = np.random.default_rng(SEED)
boot = [rng.integers(0, len(va), len(va)) for _ in range(300)]
random_value = float(np.sum(w * va_profit) / np.sum(w))   # the emails as they were actually sent
results = {"Random department (today)": {"value": random_value, "predicted": None}}
choices = {}
for name, P in [("Logistic per department", pA), ("LightGBM, one model", pB)]:
    for basis, S in [("by purchase chance", P), ("by expected profit", P * profit_per_buy.to_numpy())]:
        c = S.argmax(1); label = f"{name}, {basis}"
        choices[label] = c
        results[label] = {"value": ips(c), "predicted": float(S.max(1).mean()) if basis == "by expected profit" else None,
                          "mix": {d: float((c == j).mean()) for j, d in enumerate(DEPTS)}}
best_single = max(range(K), key=lambda j: ips(np.full(len(va), j)))
results[f"Everyone gets {DEPTS[best_single]}"] = {"value": ips(np.full(len(va), best_single)), "predicted": None}

for k, r in results.items():
    lift = r["value"] / random_value - 1
    lo, hi = None, None
    if k in choices:
        diffs = [ips(choices[k], b) - float(np.sum(w[b] * va_profit[b]) / np.sum(w[b])) for b in boot]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        r["ci_per_customer"] = [float(lo), float(hi)]
    r["lift"] = lift; r["campaign_gain"] = (r["value"] - random_value) * CAMPAIGN
    pred = f"  model said ${r['predicted']:.3f}" if r.get("predicted") else ""
    ci = f"  95% CI gain/customer [{lo:.3f}, {hi:.3f}]" if lo is not None else ""
    print(f"{k:48s} ${r['value']:.3f}/customer  lift {lift:+.1%}  10M emails {r['campaign_gain']/1e6:+.2f}M{pred}{ci}")

# Part 1 of the case: six-month LTV by weekly email frequency (Exhibit 1).
freq = {1: (0.024, 9.97, 6.32), 2: (0.034, 14.21, 6.36), 3: (0.097, 14.56, 6.70), 4: (0.171, 14.89, 7.11)}
ltv = {}
for f, (u, rs, ru) in freq.items():
    active, v = 1.0, 0.0
    for t in range(1, 7):
        active *= 1 - u
        v += (active * rs + (1 - active) * ru) * MARGIN / 1.01 ** t
    ltv[f] = v
print("six-month LTV by emails/week:", {k: round(v, 2) for k, v in ltv.items()})
save("ikea", {"trueRate": O, "margin": MARGIN, "campaign": CAMPAIGN, "profitPerBuy": profit_per_buy.to_dict(),
              "policies": results, "rounds": es.best_iteration, "ltvByFrequency": ltv})
