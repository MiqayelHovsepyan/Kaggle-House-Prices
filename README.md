# 🏠 Kaggle House Prices — Advanced Regression Techniques

A solution for the Kaggle competition
[**House Prices: Advanced Regression Techniques**](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview).
The task is to predict the final sale price of residential homes in Ames, Iowa from 79 explanatory
features. Models are scored on **RMSLE** (Root Mean Squared *Logarithmic* Error) — the error between
the logarithms of the predicted and the actual price.

---

## 📁 Project structure

```
data/
├── raw/                # Original Kaggle data (train.csv, test.csv)
└── processed/          # Cleaned, feature-selected data produced by EDA.ipynb
notebooks/
├── EDA.ipynb           # Exploratory analysis + all preprocessing / feature engineering
└── models.ipynb        # Model comparison, selection, and submission generation
submission.csv          # Final predictions for Kaggle
requirements.txt
```

---

## 🔎 What I did

The work is split into two notebooks:

1. **`EDA.ipynb` — understanding and preparing the data.** I explored the 79 raw features,
   handled missing values, removed weak/redundant columns, and selected a compact set of strong
   predictors. The cleaned, fully-numeric dataset is saved to `data/processed/` so modeling can start
   from a stable foundation.
2. **`models.ipynb` — trying many models and choosing the best.** Starting from the processed data, I
   trained **seven** regression models under one fair cross-validation scheme, compared them, and used
   the winner to produce the submission.

### Conclusions from the data (EDA)

- **Missing values.** Columns that were mostly empty were dropped — `PoolQC` (~99.5% missing),
  `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `MasVnrType`, `LotFrontage`. The rest were imputed
  (median for numeric, mode for categorical).
- **Redundancy & low signal.** Highly-correlated and near-constant columns were removed (e.g.
  `TotalBsmtSF`/`GrLivArea` overlap with other area features; `Utilities`/`Street` are almost constant).
- **Feature selection.** A Random-Forest importance ranking kept the strongest predictors. The final
  model set is **13 numeric features**: `OverallQual`, `1stFlrSF`, `2ndFlrSF`, `GarageArea`,
  `BsmtFinSF1`, `LotArea`, `MasVnrArea`, `OpenPorchSF`, `WoodDeckSF`, `YearRemodAdd`, `BsmtUnfSF`,
  `YearBuilt`, `GarageYrBlt`. Overall quality, size, and age dominate — `OverallQual` alone accounts
  for ~47% of the winning model's importance.
- **Skewed target.** `SalePrice` is strongly right-skewed. Since Kaggle scores on the log of the price,
  the modeling notebook trains on **`log1p(SalePrice)`** and converts predictions back with `expm1`.
  This both matches the leaderboard metric and stabilizes the error.

---

## 🤖 What I tried (models)

All models share the **same** 5-fold cross-validation split and are evaluated on **RMSLE** (RMSE on
the log target) and **R²**. Linear models run inside a `StandardScaler` pipeline; tree/boosting models
are scale-invariant. XGBoost and LightGBM are lightly grid-searched.

| Rank | Model | CV RMSLE | CV R² |
|----:|-------|:--------:|:-----:|
| 🥇 1 | **XGBoost** | **0.1461** | **0.863** |
| 2 | LightGBM | 0.1480 | 0.859 |
| 3 | Gradient Boosting | 0.1525 | 0.850 |
| 4 | Random Forest | 0.1578 | 0.838 |
| 5 | Lasso | 0.1680 | 0.809 |
| 6 | Linear Regression | 0.1700 | 0.800 |
| 7 | Ridge | 0.1711 | 0.798 |

---

## 🏆 Final result

**XGBoost** is the best model (CV **RMSLE ≈ 0.146**, **R² ≈ 0.86**), tuned to
`learning_rate=0.03, max_depth=4, n_estimators=300, reg_lambda=5.0`. It is retrained on the full
training set and used to generate `submission.csv`.

- Previous single-model submission (XGBoost on the raw price): **Kaggle RMSLE 0.15452**.
- Training on the log target (matching the competition metric) brings the cross-validated error down to
  **≈ 0.146** — a clear improvement.

> The Kaggle leaderboard score is obtained by uploading `submission.csv` on the competition page.

---

## 🧠 Why some models performed better than others

- **Gradient-boosted trees win (XGBoost, LightGBM, Gradient Boosting).** Price depends on the features
  in a *non-linear*, *interacting* way — the value of extra living area depends on quality and era, and
  quality itself has an accelerating effect on price. Boosting builds many small trees, each correcting
  the previous ones' residuals, so it captures these curves and interactions directly. XGBoost edges
  ahead thanks to its built-in regularization and row/column subsampling, which prevent overfitting on
  the ~1,460 training rows.
- **Random Forest is solid but slightly behind boosting.** Averaging independent trees (bagging)
  reduces variance, but it doesn't keep chasing the residual bias the way sequential boosting does.
- **Linear models underperform (Linear, Ridge, Lasso — RMSLE ≈ 0.17).** They can only fit a single
  straight, additive surface, so they *underfit* the non-linear relationships above. Regularization
  (Ridge's L2, Lasso's L1) only controls variance — it can't add the missing non-linearity, which is
  why their scores barely differ. Lasso is marginally best of the three because its L1 penalty trims
  the noisiest features.

---

## ⚙️ How to run

```bash
pip install -r requirements.txt
# 1) (optional) regenerate processed data
jupyter notebook notebooks/EDA.ipynb
# 2) compare models, pick the best, write submission.csv
jupyter notebook notebooks/models.ipynb
```

`notebooks/models.ipynb` reads `data/processed/`, evaluates all seven models, and writes
`submission.csv` from the best one.

---

## 🧑‍💻 Author

**Miqayel Hovsepyan** — Yerevan State University, Informatics and Applied Mathematics;
AI track @ Picsart Academy.

Competition: <https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview>
