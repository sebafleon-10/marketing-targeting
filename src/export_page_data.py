"""Compact data for the portfolio page. Usage: export_page_data.py OUT.json (needs results/ and data/raw/IKEA.csv)."""
import json, sys
import numpy as np, pandas as pd

dd, st, ik = (json.load(open(f"results/{n}.json")) for n in ("doordash", "stumptown", "ikea"))
pol = lambda res, names: [next(p for p in res["policies"] if p["policy"] == n) for n in names]

d = pd.read_csv("data/raw/IKEA.csv"); va = d[d.training == 0].copy()
va["w"] = np.where(va.buy == 1, ik["trueRate"] / 0.5, (1 - ik["trueRate"]) / 0.5)
wmean = lambda x: float(np.average(x.buy, weights=x.w))
by_dept = va.groupby("message").apply(wmean, include_groups=False).sort_values()
va["band"] = pd.qcut(va.income, 3, labels=["Lower income", "Middle income", "Higher income"])
by_income = va.groupby("band", observed=True).apply(wmean, include_groups=False)

ip = ik["policies"]
pers = ip["Logistic per department, by expected profit"]
single_key = next(k for k in ip if k.startswith("Everyone gets"))

# Bootstrap CI for the single-department policy, same IPS logic as the main script.
P = (va.buy * va.order_size * ik["margin"]).to_numpy(); W = va.w.to_numpy(); M = va.message.to_numpy()
top = single_key.replace("Everyone gets ", ""); rng = np.random.default_rng(42); diffs = []
for _ in range(300):
    b = rng.integers(0, len(va), len(va))
    base = np.sum(W[b] * P[b]) / np.sum(W[b])
    diffs.append(np.sum(W[b] * (M[b] == top) * 7 * P[b]) / np.sum(W[b]) - base)
single_ci = np.percentile(diffs, [2.5, 97.5]).tolist()

out = {
  "doordash": {"cost": dd["cost"], "breakeven": dd["breakeven"], "responseRate": dd["responseRate"], "base": dd["base"],
               "policies": [{"label": l, "profit": p["profit"], "roi": p["roi"], "share": p["targeted_share"]}
                            for l, p in zip(["Send to everyone", "RFM segments", "Logistic model"], pol(dd, ["Send to everyone", "RFM cells", "Logistic regression"]))],
               "gbm": pol(dd, ["LightGBM"])[0]["profit"], "robust": dd["robust"],
               "gains": {"RFM segments": dd["gains"]["RFM cells"], "Logistic model": dd["gains"]["Logistic regression"]}},
  "stumptown": {"cost": st["offerCost"], "win": st["winValue"], "breakeven": st["breakeven"], "conversion": st["conversion"], "remaining": st["remaining"],
                "policies": [{"label": l, "profit": p["profit"], "roi": p["roi"], "share": p["targeted_share"]}
                             for l, p in zip(["Sample everyone", "Logistic model", "Gradient boosting"], pol(st, ["Sample everyone", "Logistic regression", "LightGBM"]))],
                "deciles": next(x["rates"] for x in st["deciles"] if x["model"] == "Logistic regression"),
                "coef": st["seasonalCoef"], "auc": st["auc"]},
  "ikea": {"trueRate": ik["trueRate"], "campaign": ik["campaign"],
           "policies": [{"label": "Random emails (today)", "value": ip["Random department (today)"]["value"], "ci": None},
                        {"label": "Personalized model", "value": pers["value"], "ci": pers["ci_per_customer"]},
                        {"label": f"Everyone gets {top.replace('KitchenDining','Kitchen').replace('HomeDecor','Home Decor').replace('LivingRoom','Living Room')}", "value": ip[single_key]["value"], "ci": single_ci}],
           "promised": pers["predicted"],
           "byDept": {k: v for k, v in by_dept.items()}, "byIncome": {k: v for k, v in by_income.items()},
           "ltv": ik["ltvByFrequency"]},
}
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"), default=float)
print(json.dumps({k: [round(p.get("profit", p.get("value", 0)), 3) for p in v["policies"]] for k, v in out.items()}))
print("byDept", {k: round(v, 5) for k, v in by_dept.items()}); print("byIncome", {k: round(v, 5) for k, v in by_income.items()}); print("single CI", np.round(single_ci, 3))
