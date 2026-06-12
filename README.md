# Montgomery County Warehouse & Retail Sales Predictor

An end-to-end modular machine learning pipeline that predicts monthly retail sales for liquor, wine, beer, and non-alcohol products in Montgomery County, MD. Built on real government data covering 307K+ records across 2017–2020, with a Streamlit app for interactive predictions.

---

## Demo

> Run locally with `streamlit run app.py` after retraining (see setup below).

---

## Project Structure

```
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Data loading, cleaning, train/test split
│   │   ├── data_transformation.py  # Feature engineering, target encoding, preprocessing pipeline
│   │   └── model_trainer.py        # Model training, hyperparameter tuning, model selection
│   ├── pipeline/
│   │   └── predict_pipeline.py     # Inference pipeline for new predictions
│   ├── exception.py                # Custom exception with traceback detail
│   ├── logger.py                   # Timestamped logging to /logs
│   └── utils.py                    # save/load object, evaluate_models with GridSearchCV
├── data/
│   └── Warehouse_and_Retail_Sales.csv  # Raw dataset (not tracked in git)
├── artifacts/                      # Generated at runtime (not tracked in git)
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   └── test.csv
├── notebook/
│   └── EDA.ipynb                   # Exploratory data analysis
├── app.py                          # Streamlit app
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [Montgomery County Open Data — Warehouse and Retail Sales](https://data.montgomerycountymd.gov/d/v76h-r7br)

- 307,645 records of monthly item-level sales across suppliers and product categories
- Updated monthly by Montgomery County, MD
- Covers LIQUOR, WINE, BEER, NON-ALCOHOL, KEGS, and store supplies

**After cleaning:**
- Dropped KEGS, DUNNAGE, STR_SUPPLIES, REF (zero or irrelevant retail sales)
- Removed null and "Default" supplier rows (~430 records)
- Filtered to non-zero retail sales only
- Dropped RETAIL TRANSFERS (0.96 correlation with target — data leakage)
- Final training shape: 185,490 rows × 6 features

---

## Approach

### Features
| Feature | Type | Notes |
|---|---|---|
| YEAR | Numerical | 2017–2020 |
| MONTH | Numerical | 1–12 |
| WAREHOUSE SALES | Numerical | Strongest continuous predictor |
| ITEM CODE | Numerical | Target-encoded from training set |
| ITEM TYPE | Categorical | LIQUOR / WINE / BEER / NON-ALCOHOL |
| SUPPLIER | Categorical | Top 30 by volume; rest collapsed to OTHER |

### Target
`RETAIL SALES` — log-transformed (`np.log1p`) during training to handle right skew; predictions are inverse-transformed (`np.expm1`) before display.

### Pipeline
- **Numerical:** median imputation → standard scaling
- **Categorical:** most-frequent imputation → one-hot encoding → standard scaling
- **ITEM CODE:** target encoding using training data only (mean retail sales per item code)

### Models Evaluated
GridSearchCV (3-fold CV) across 7 models:

| Model | R² |
|---|---|
| **Random Forest** | **0.8875** |
| XGBoost | 0.8749 |
| CatBoost | 0.8720 |
| Gradient Boosting | 0.8682 |
| AdaBoost | 0.8336 |
| Decision Tree | 0.8175 |
| Linear Regression | 0.4981 |

Random Forest selected as best model. Linear Regression's low score confirms non-linear relationships in the data.

---

## Setup

```bash
git clone https://github.com/sejalnimkar/DS-project.git
cd DS-project
pip install -r requirements.txt
```

Download the dataset from [Montgomery County Open Data](https://data.montgomerycountymd.gov/d/v76h-r7br) and place it at:
```
data/Warehouse_and_Retail_Sales.csv
```

**Retrain the model:**
```bash
python src/components/data_ingestion.py
```
This runs the full pipeline — ingestion, transformation, hyperparameter tuning, and saves `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

**Run the app:**
```bash
streamlit run app.py
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
catboost
streamlit
```

---

## Key Design Decisions

- **Log-transform target:** Raw retail sales are heavily right-skewed with a spike at zero. Filtering to non-zero rows and applying `log1p` significantly improved model performance across all tree-based models.
- **Target encoding for ITEM CODE:** ~10K unique item codes make one-hot encoding infeasible. Target encoding (mean retail sales per item) captures item-level signal without cardinality explosion.
- **RETAIL TRANSFERS dropped:** 0.96 correlation with the target makes it a leakage feature — it would not be available at true prediction time.
- **Supplier collapsed to top 30:** Reduces cardinality while preserving signal from the highest-volume distributors.
