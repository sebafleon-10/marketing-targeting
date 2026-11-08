# Who gets the offer? Three targeting decisions, scored honestly

Sebastian Leon. A rebuild of three BUSA 390 Marketing Analytics cases (DePauw) with out-of-sample evaluation.

The cases are **fictionalized teaching cases with simulated data**, written for the course. The company names are the case settings, not sources of real company data. The case documents and datasets belong to the course and are not included here; only code and aggregate results are.

## 1. DoorDash: should we send the push?
70,000 customers got a free-delivery push. Each push costs about $0.96 in lost future value (opt-outs). Targets are chosen on one half of the data and scored on the other, scaled to the 36.93M remaining customers.

| Policy | Profit | ROI |
|---|---|---|
| Send to everyone | -$1.27M | -3.6% |
| RFM cells above breakeven | +$4.11M | 26.6% |
| Logistic regression | +$4.55M | 30.7% |
| LightGBM | +$4.53M | 32.5% |

Over 20 random splits, logistic beats RFM every time (mean +$0.51M, sd $0.26M); LightGBM adds nothing over logistic. No control group was held out, so this targets likely responders, not people the push changes.

## 2. Stumptown: who gets a free sample?
A sample costs $20; a 20-lb reorder is worth $270, so breakeven is 7.4 percent. Scores are out-of-fold from 5-fold cross-validation, scaled to the 45,000 remaining accounts.

| Policy | Profit | ROI |
|---|---|---|
| Sample everyone | $159K | 17.7% |
| Logistic regression | $458K | 137% |
| LightGBM | $462K | 148% |

Seasonal single-origin purchases look like a strong signal alone (logit coefficient 0.25) and nearly vanish (0.002) once spend and the other histories are controlled for.

## 3. IKEA: which email should each customer get?
Each customer's last email came from a department chosen at random (1 in 7), so a personalization policy can be scored with inverse propensity weighting on the validation split. Buyers were oversampled to 50 percent; weights restore the true 0.84 percent.

| Policy | Profit per customer | vs random |
|---|---|---|
| Random department (today) | $0.592 | |
| Per-department logistic, by expected profit | $0.614 | +3.7% (95% CI includes 0) |
| One LightGBM with department as input | $0.592 | +0.0% |
| Everyone gets Bedroom | $0.619 | +4.6% (95% CI includes 0) |

The per-department model predicts $4.04 per customer for itself; the random assignment says $0.61. Department barely moves purchase rates (0.80 to 0.89 percent across all seven); income moves them about 25 times. Who gets emailed matters far more than which email.

Six-month LTV by weekly emails (Exhibit 1, compounding unsubscribes, 1 percent monthly discount, 35 percent margin): 1/week $19.63, **2/week $27.05**, 3/week $24.95, 4/week $23.08.

## Run it
Place the course CSVs in `data/raw/` (DoorDash.csv, StumptownWholesale.csv, IKEA.csv), then:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
brew install libomp   # macOS, for LightGBM
.venv/bin/python src/doordash.py
.venv/bin/python src/stumptown.py
.venv/bin/python src/ikea.py
```
