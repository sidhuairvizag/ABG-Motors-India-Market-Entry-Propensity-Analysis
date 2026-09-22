# ABG Motors — India Market Entry Propensity Analysis

Data-driven market-entry analysis that predicts vehicle purchase propensity for ABG Motors in India.

Train an interpretable purchase model on labeled Japanese customers, score a 70,000-customer Indian sample, stress-test the forecast for Japan–India income scale shift, and turn the scores into a go / no-go recommendation plus a CRM targeting list.

This project is written as a junior-analyst case study: business question first, method second, every number with a denominator, and explicit limits on what the model can claim.

---

## 1. Business question

ABG Motors is deciding whether to enter India. Leadership set a sample hurdle of **12,000 expected vehicle sales** on a 70,000-customer Indian file.

Japan already has labeled purchase outcomes. India has customer attributes and last-maintenance dates, but **no purchase label**.

The job is therefore not “build the most complex model.” It is:

1. Learn which attributes are associated with purchase in Japan.
2. Transfer that relationship to India with the same feature rules.
3. Produce **expected sales = sum of predicted probabilities**, not only a 0/1 class count.
4. Check whether the conclusion survives a known domain shift: Indian incomes sit on a different scale from Japanese incomes.
5. Tell a commercial team who to call first.

### Decision from this sample

**Recommendation: PILOT, THEN ENTER.**

Do not treat the unadjusted 60,318 figure as a national India forecast.

| Scenario                         | Expected sales | Multiple of 12,000 |
|----------------------------------|----------------|--------------------|
| Logistic, unadjusted             | 60,318         | 5.0x               |
| Logistic, income-aligned         | 40,987         | 3.4x               |

- Both stay above the 12,000 hurdle. The cautious planning number is **40,987**.
- Predicted buyers at p >= 0.50 (unadjusted logistic): **67,362**.
- Mean unadjusted probability: **0.86**.

Action: run a city-level pilot and CRM offers on high-propensity, high car-age customers. Scale only if live conversion matches the scored ranking.

---

## 2. Why this is a transfer problem

| Market | Rows   | Label              | Purchase rate |
|--------|--------|--------------------|---------------|
| Japan  | 40,000 | PURCHASE (labeled) | 57.6% (23,031 buyers) |
| India  | 70,000 | none (scoring only)| model scores only     |

A model trained in Market A and applied to Market B is only as good as the feature mapping and the assumption that the relationship still holds. Two design choices matter more than leaderboard AUC:

- India `AGE_CAR` is derived from last maintenance date using a fixed analysis date of **1 July 2019**, then binned with the same four segments used in Japan.
- Raw India `ANN_INCOME` is not treated as the same unit as Japan income. A percentile-mapped **income-aligned** score is produced as a sensitivity case.

Expected sales is the sum of probabilities. That is the right quantity for a volume hurdle. A headcount of customers with p >= 0.50 is reported separately and is not the same thing.

---

## 3. Repository map

```text
ABG-Motors-India-Market-Entry-Propensity-Analysis/
├── README.md
├── requirements.txt
├── data/
│   ├── raw data/
│   │   ├── JPN Data.xlsx                  # Japan training market (40,000 rows, PURCHASE label)
│   │   └── IN_Data.xlsx                   # India scoring market (70,000 rows, no label)
│   └── processed data/
│       ├── Indian_Scored_Customers.csv    # scored India file (logistic + XGBoost, raw + income-aligned)
│       ├── Logistic_Coefficients.csv      # primary-model log-odds and odds ratios
│       └── Model_Performance_Comparison.csv
├── notebook code/
│   └── ABG-Motors-India-Market-Entry-Propensity-Analysis.ipynb
├── tableau analysis/
│   └── ABG Motors – India Market Entry Decision.twb
└── dashboard images/
    ├── Dashboard - 1 ABG Motors – India Market Entry Decision.jpg
    ├── Dashboard - 2 ABG Motors - Model Performance and Drivers.jpg.jpg
    ├── Dashboard - 3 ABG Motors - Indian Opportunity & CRM.jpg
    └── Dashboard - 4 ABG Motors – Japan Training Market What Drives Purchase.jpg
```

---

## 4. Data dictionary

### Japan (`JPN Data.xlsx`)

| Column      | Description |
|-------------|-------------|
| ID          | Customer identifier |
| CURR_AGE    | Current age of the customer |
| GENDER      | M / F |
| ANN_INCOME  | Annual income (Japan scale) |
| AGE_CAR     | Age of current car in days |
| PURCHASE    | 1 = purchased, 0 = did not (training label) |

### India (`IN_Data.xlsx`)

| Column      | Description |
|-------------|-------------|
| ID          | Customer identifier |
| CURR_AGE    | Current age of the customer |
| GENDER      | M / F |
| ANN_INCOME  | Annual income (India scale — not interchangeable with Japan) |
| DT_MAINT    | Last maintenance date |

### Engineered on both markets

| Column      | Rule |
|-------------|------|
| AGE_CAR (India) | (2019-07-01) minus DT_MAINT, in days |
| AGE_SEG     | 1: <200 days; 2: 200–360 days; 3: 360–500 days; 4: >500 days |
| GENDER_ENC  | 1 if M, else 0 |

### Scored India (`Indian_Scored_Customers.csv`)

- `proba_logreg` / `pred_logreg`
- `proba_logreg_aligned` / `pred_logreg_aligned`
- `proba_xgb` / `pred_xgb`
- `proba_xgb_aligned` / `pred_xgb_aligned`

`pred_*` uses threshold 0.50 unless a sensitivity table says otherwise.

---

## 5. Method

### Step 0 — Setup

pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn, plotly, openpyxl.

### Step 1 — Load and inspect

Confirm shapes, columns, and that India has no target. India is a scoring problem, not a supervised India model.

### Step 2 — Quality checks

Missing-value audit on both files. Descriptive statistics. Group-by aggregates in the same shape a SQL analyst would write: counts, purchase rate, mean income, mean age by gender and by AGE_SEG.

### Step 3 — Feature rules (locked to the brief)

Same AGE_SEG bins on both markets. India car age is computed from `DT_MAINT` against 1 July 2019 so the two markets share a definition.

### Step 4 — Exploratory analysis on the labeled market

Japan purchase rate by AGE_SEG is the dominant pattern:

| AGE_SEG | Car age        | Purchase rate | Customers |
|---------|----------------|---------------|-----------|
| 1       | <200 days      | 36.4%         | 6,459     |
| 2       | 200–360 days   | 42.4%         | 16,545    |
| 3       | 360–500 days   | 78.7%         | 11,697    |
| 4       | >500 days      | 84.0%         | 5,299     |

Gender gap is small (M 59.2%, F 55.5%). Mean car age in Japan is 359.1 days.

Income histograms for Japan vs India are plotted **before** any model is trained. The scale mismatch is a first-class risk, not an afterthought.

### Step 5 — Modeling matrix

Features: `CURR_AGE`, `GENDER_ENC`, `ANN_INCOME`, `AGE_SEG` (dummy-encoded, first level dropped).

`CURR_AGE` and `ANN_INCOME` are standardized with a scaler **fit on Japan only**.

Stratified 75/25 train/test split, `random_state=42`, so class balance is preserved in the holdout.

### Step 6 — Two models, different jobs

- **Primary:** Logistic Regression (`max_iter=2000`, `C=1.0`). Chosen for coefficients a commercial team can read.
- **Comparison:** XGBoost (`n_estimators=200`, `max_depth=5`, `learning_rate=0.06`, `subsample=0.85`, `colsample_bytree=0.8`, `min_child_weight=5`). Used to check that the ranking is not an artifact of a linear specification.

No India labels are used in training. That would be leakage, and India has no labels.

### Step 7 — Holdout evaluation (Japan test set)

| Model                         | Accuracy | Precision | Recall | F1     | ROC-AUC |
|-------------------------------|----------|-----------|--------|--------|---------|
| Logistic Regression (primary) | 68.64%   | 74.42%    | 69.38% | 71.81% | 0.7600  |
| XGBoost (comparison)          | 69.79%   | 74.78%    | 71.73% | 73.22% | 0.7812  |

XGBoost is slightly stronger on ranking. Logistic is close enough and produces odds ratios. The decision layer uses logistic. That is a product choice, not a claim that logistic “wins.”

### Step 8 — Score India

Apply Japan dummy columns, the Japan scaler, and both fitted models. Report:

- expected sales = sum(probability)
- predicted buyers = count(probability >= 0.50)
- mean probability
- threshold sweep at 0.50 / 0.60 / 0.70 / 0.80

### Step 9 — Domain-shift sensitivity

For each Indian customer, replace raw `ANN_INCOME` with the Japanese income at the same percentile, transform with the Japan scaler, and rescore.

This is not a claim that the two countries have the same cost of living. It is a check: if income is only a relative rank, does the sample still clear 12,000?

### Step 10 — Stakeholder views

1. Market-entry decision — expected sales vs 12,000, threshold sensitivity, written recommendation.
2. Model performance and drivers — AUC, odds ratios, AGE_SEG probabilities.
3. Indian opportunity and CRM — segment volumes, expected sales by AGE_SEG, probability distribution.
4. Japan training market — base rates the model is allowed to learn from.

---

## 6. Primary model drivers (logistic odds ratios)

| Feature     | Odds ratio | How to read it |
|-------------|------------|----------------|
| AGE_SEG_4   | 10.10      | vs AGE_SEG 1, holding other features |
| AGE_SEG_3   | 6.71       | vs AGE_SEG 1 |
| ANN_INCOME  | 1.55       | per 1 SD on the Japan-scaled income feature |
| AGE_SEG_2   | 1.32       | vs AGE_SEG 1 |
| GENDER_ENC  | 1.23       | male vs female |
| CURR_AGE    | 0.87       | per 1 SD; older customers slightly less likely, all else equal |

The commercial story is **car age**, not gender. Customers whose current vehicle is older than about a year are the replacement pool. That pattern is visible in raw Japan rates before the model is fit, which is the check you want on an odds-ratio table.

---

## 7. India sample results (logistic unless noted)

Sample size: **70,000**

### Unadjusted logistic

- Expected sales: **60,318**
- Predicted buyers p >= 0.50: **67,362**
- Mean probability: **86.17%**

Threshold sensitivity (buyer count):

| Cutoff   | Buyers |
|----------|--------|
| p >= 0.50 | 67,362 |
| p >= 0.60 | 64,988 |
| p >= 0.70 | 60,399 |
| p >= 0.80 | 52,265 |

### Income-aligned logistic

- Expected sales: **40,987**

### India customers by AGE_SEG

| AGE_SEG | Customers |
|---------|-----------|
| 1 (<200 days)    | 18,496 |
| 2 (200–360 days) | 18,994 |
| 3 (360–500 days) | 18,691 |
| 4 (>500 days)    | 13,819 |

### Expected sales by AGE_SEG (unadjusted logistic)

| AGE_SEG | Expected sales |
|---------|----------------|
| 1 | 14,072 |
| 2 | 15,257 |
| 3 | 17,683 |
| 4 | 13,306 |

AGE_SEG 3+4 combined: **32,510 customers**, **30,989 expected sales**.

### Average predicted probability by AGE_SEG (unadjusted)

| AGE_SEG | Mean probability |
|---------|------------------|
| 1 | 76.08% |
| 2 | 80.33% |
| 3 | 94.61% |
| 4 | 96.29% |

CRM rule: rank the file by probability, then concentrate service-to-sales, financing, and trade-in offers on AGE_SEG 3 and 4.

32,510 is the count of customers in those two segments. It is **not** the 67,362 buyer flag.

---

## 8. Recommendation logic

- Clear the hurdle on more than one scoring rule.
- Stay below a national-forecast claim.
- Separate “expected sales” from “customers above a cutoff.”
- Name the pilot unit (city), the watch metric (live conversion vs score decile), and the risks that would stop a rollout.

### Risks (kept on the decision dashboard on purpose)

1. Income scale difference between Japan and India.
2. The file is one sample, not a national census.
3. The model is correlational. Run CRM A/B tests before national spend.

A 0.76 ROC-AUC on Japan holdout means the model ranks buyers above non-buyers better than chance. It does **not** mean 86% of Indian customers will buy. The 86% figure is an unadjusted mean score and is expected to be optimistic when raw India income is pushed through a Japan-fitted income coefficient. That is why the aligned number exists.

---

## 9. How to run

pip install -r requirements.txt
Run notebook/ABG-Motors-India-Market-Entry-Propensity-Analysis.ipynb with JPN Data.xlsx and IN_Data.xlsx in the notebook working folder. 

requirements.txt

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- plotly
- openpyxl


---
