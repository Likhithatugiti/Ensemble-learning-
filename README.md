# Gridlock — Traffic Demand Forecasting (Flipkart GRiD)

Ensemble learning solution built for the Flipkart GRiD hackathon. The notebook (`Gridlock.ipynb`) predicts road traffic demand per location (geohash) and time slot by combining gradient-boosted tree models into an ensemble, rather than relying on a single estimator.

## Problem Statement

Given road-level and environmental attributes (location, road type, number of lanes, large vehicle presence, landmarks, weather, temperature) at a given day and timestamp, predict `demand` — the traffic load for that geohash at that time.

## Model Architecture

### 1. Feature Engineering
- **Temporal features**: timestamp (`H:MM`) parsed into `hour`, `minute`, `time_in_minutes`; cyclic encoding via `hour_sin`/`hour_cos`; day-of-week proxy (`day_mod7`) with `day_sin`/`day_cos` cyclic encoding.
- **Spatial / geohash features**: per-geohash aggregate demand stats (`mean`, `median`, `std`, `count`) computed on train and joined into both train/test; geohash truncated to 3/4/5-character prefixes to capture demand at multiple spatial resolutions, label-encoded.
- **Spatio-temporal interaction**: average demand per `(geohash, hour)` pair (`geo_hour_mean`), backfilled with the geohash-level mean for unseen combinations.
- **Categorical handling**: `RoadType`, `LargeVehicles`, `Landmarks`, `Weather` filled with `'Unknown'`/`'missing'` and either label-encoded (for LightGBM/XGBoost) or passed natively as categoricals (for CatBoost).
- **Numeric cleanup**: `Temperature` missing values imputed using the median temperature per `Weather` type (falling back to global median).

### 2. Train/Validation Split
A **chronological split**, not a random one: records from `day == 49` at or after 1:30 AM are held out as the validation set, with everything before that as training. This mimics forecasting into the future rather than leaking nearby timestamps into validation.

### 3. Base Models
Three gradient-boosting regressors are trained independently:

| Model | Library | Key config |
|---|---|---|
| CatBoost | `CatBoostRegressor` | `iterations=1000`, `lr=0.05`, `depth=6`, RMSE loss, native categorical features, early stopping (50 rounds) |
| LightGBM | `LGBMRegressor` | `objective='regression_l1'` (MAE), `n_estimators=1000`, `lr=0.05`, `num_leaves=31`, `subsample/colsample=0.8`, L1/L2 regularization, early stopping |
| XGBoost | `XGBRegressor` | `objective='reg:squarederror'`, `n_estimators=1000`, `lr=0.05`, `max_depth=6`, `subsample/colsample=0.8`, early stopping |

CatBoost additionally goes through a `GridSearchCV` hyperparameter sweep (`iterations`, `learning_rate`, `depth`) with 3-fold CV on RMSE.

All models predict non-negative demand via `np.clip(preds, 0, None)`.

### 4. Model Selection & Ensembling

We trained and evaluated three boosted-tree models — CatBoost, LightGBM, and XGBoost — before deciding on the final ensemble.

**Why CatBoost was left out of the final ensemble**: CatBoost was trained on a smaller feature set (`minutes`, `NumberofLanes`, `Temperature`, the geo-level demand stats, and the raw categoricals) and didn't include the richer engineered signals — cyclic time encoding (`hour_sin`/`hour_cos`, `day_sin`/`day_cos`), the `geo_hour_mean` spatio-temporal interaction, or the multi-resolution geohash prefixes (3/4/5-character) — that LightGBM and XGBoost were given. Even after a dedicated `GridSearchCV` hyperparameter sweep, CatBoost's validation RMSE/MAE/R² came in weaker than the other two models on the held-out chronological split, so it was kept as a benchmark/comparison run rather than folded into the final blend.

**Why LightGBM + XGBoost were ensembled**: both were trained on the identical, fuller feature set, so their errors are driven by genuinely different model mechanics rather than missing information — LightGBM was tuned with an MAE objective (`regression_l1`) while XGBoost used a squared-error objective, so they tend to make different kinds of mistakes on the same rows. Averaging the two smooths out model-specific noise/variance without needing a learned meta-model, and the validation metrics for the blend came out more stable than either model alone:

```
ensemble_prediction = (lightgbm_prediction + xgboost_prediction) / 2
```

This is applied to both the validation set (for scoring) and the test set (for the final submission).

### 5. Evaluation
Each model — and the final ensemble — is scored on RMSE, MAE, and R² against the chronological validation split.

### 6. Output
Final predictions are written to `submission.csv` with `Index` and `demand` columns.

## Pipeline Summary

```
raw train/test CSVs
   → time + cyclic feature extraction
   → geohash-level & geohash×hour demand aggregates
   → multi-resolution geohash prefixes + categorical encoding
   → chronological train/validation split (day 49, ≥1:30 AM held out)
   → train CatBoost / LightGBM / XGBoost (+ CatBoost grid search)
   → average LightGBM + XGBoost predictions
   → clip to non-negative
   → submission.csv
```

## Repository Structure

```
Ensemble-learning-/
└── Gridlock.ipynb   # End-to-end notebook: data prep, feature engineering, model training, ensembling, evaluation
```

## Getting Started

### Requirements

```
pip install numpy pandas scikit-learn catboost lightgbm xgboost
```

### Running

```bash
git clone https://github.com/Likhithatugiti/Ensemble-learning-.git
cd Ensemble-learning-
jupyter notebook Gridlock.ipynb
```

Run all cells top to bottom. Place the competition dataset (train/test CSVs) in the same directory, or update the file paths at the top of the notebook to point to your local data.

## Results

_Add final leaderboard score / key metrics here (e.g. RMSE, MAE) once finalized._

## Team / Context

Built for the Flipkart GRiD hackathon as a spatio-temporal demand forecasting solution using an ensemble learning approach.

## License

Add a license if you intend to open-source this (MIT is a common default for hackathon repos).
