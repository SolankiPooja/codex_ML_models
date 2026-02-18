# Incentive Recommendation ML Pipeline

This project provides an end-to-end Python workflow for predicting an **ideal incentive program for property owners** using:

- Raw Incentive Data
- Raw Property Data
- User Behavior Data

---

## End-to-End System Workflow (Step-by-Step)

### 1) Data Sources
Collect three raw datasets:

1. **Raw Incentive Data** (program catalog, amount, terms, etc.)
2. **Raw Property Data** (owner and property attributes)
3. **User Behavior Data** (engagement history + observed preferred program)

### 2) Data Cleaning
Each dataset is validated and cleaned:

- Required column checks
- Duplicate removal
- String normalization (trim/standardize text)
- Missing value handling (median for numeric, mode for categorical)

### 3) Feature Engineering
Datasets are merged and transformed into model-ready features:

- Owner-property join on `owner_id` + `property_id`
- Incentive landscape context features (global avg/max/min amount, total program count)
- Owner behavior aggregates (avg/max engagement, interaction count)
- Derived interaction key (`owner_property_interaction`)

### 4) Training Dataset Creation
Construct supervised training matrix:

- **Features (X):** engineered numeric + categorical attributes
- **Target (y):** `ideal_incentive_program`
- Persist selected feature columns for inference parity

### 5) Model Training
Train classification pipeline:

- Preprocessing with `ColumnTransformer`
  - Numeric: `StandardScaler`
  - Categorical: `OneHotEncoder(handle_unknown="ignore")`
- Model: `RandomForestClassifier`

### 6) Evaluation
Measure quality on holdout split:

- Accuracy
- Per-class precision/recall/F1 via classification report
- Class distribution + train/test row counts

### 7) Model Packaging
Export production artifacts:

- `artifacts/incentive_recommender.joblib` (pipeline + metadata)
- `artifacts/metrics.json`

### 8) API Deployment
Deploy FastAPI app for real-time recommendations:

- Startup loads the model artifact
- `/health` for service health
- `/recommend` for inference
- `Dockerfile` + `render.yaml` support container/cloud deploys

### 9) Real-Time Incentive Recommendations
Client sends JSON feature payload → API returns:

- `recommended_incentive_program`
- `class_probabilities` (when available)

---

## Colorful Workflow Diagram

```mermaid
flowchart TD
    A[📦 Raw Incentive Data]:::source --> D[🧹 Data Cleaning]:::process
    B[🏠 Raw Property Data]:::source --> D
    C[📲 User Behavior Data]:::source --> D

    D --> E[🧠 Feature Engineering]:::feature
    E --> F[🗂️ Training Dataset Creation]:::dataset
    F --> G[🤖 Model Training]:::train
    G --> H[📊 Evaluation]:::eval
    H --> I[🚀 Model Deployment \(FastAPI\)]:::deploy
    I --> J[⚡ Real-Time Incentive Recommendations]:::serve

    classDef source fill:#4fc3f7,stroke:#0277bd,color:#0d2538,stroke-width:2px;
    classDef process fill:#81c784,stroke:#2e7d32,color:#0f2e14,stroke-width:2px;
    classDef feature fill:#ba68c8,stroke:#6a1b9a,color:#2b1033,stroke-width:2px;
    classDef dataset fill:#ffd54f,stroke:#f9a825,color:#3e2d07,stroke-width:2px;
    classDef train fill:#ff8a65,stroke:#d84315,color:#3a1208,stroke-width:2px;
    classDef eval fill:#90a4ae,stroke:#455a64,color:#152026,stroke-width:2px;
    classDef deploy fill:#64b5f6,stroke:#1565c0,color:#0f2338,stroke-width:2px;
    classDef serve fill:#f06292,stroke:#ad1457,color:#3a1021,stroke-width:2px;
```

---

## Project Structure

```text
src/incentive_model/
  data_pipeline.py   # cleaning + feature engineering + training set creation
  train.py           # model training + evaluation + artifact creation
  api.py             # FastAPI service for real-time predictions
Dockerfile           # containerized API deployment
render.yaml          # Render deployment config
```

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Expected Input Files

Prepare CSV files with at least these columns:

- `incentive_data.csv`:
  - `incentive_program`
  - `incentive_amount`
- `property_data.csv`:
  - `property_id`
  - `owner_id`
  - (plus any other property features)
- `behavior_data.csv`:
  - `owner_id`
  - `property_id`
  - `engagement_score`
  - `ideal_incentive_program` (target label)
  - (plus any other behavior features)

## 3) Train + Evaluate

```bash
PYTHONPATH=src python -m incentive_model.train \
  --incentive-data data/incentive_data.csv \
  --property-data data/property_data.csv \
  --behavior-data data/behavior_data.csv \
  --output-dir artifacts
```

Output artifacts:

- `artifacts/incentive_recommender.joblib` (trained pipeline + metadata)
- `artifacts/metrics.json` (accuracy and classification report)

## 4) Deploy API locally

```bash
PYTHONPATH=src uvicorn incentive_model.api:app --host 0.0.0.0 --port 8000
```

Interactive API docs:

- http://localhost:8000/docs

Health check:

```bash
curl http://localhost:8000/health
```

Prediction request:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "owner_id": "O-102",
      "property_id": "P-21",
      "engagement_score": 0.82,
      "city": "Dallas",
      "property_type": "Single Family",
      "global_avg_incentive_amount": 1800,
      "global_max_incentive_amount": 2400,
      "global_min_incentive_amount": 750,
      "available_program_count": 5,
      "owner_avg_engagement": 0.7,
      "owner_max_engagement": 0.95,
      "owner_interaction_count": 11
    }
  }'
```

`owner_property_interaction` is auto-generated by the API if `owner_id` and `property_id` are provided.

## 5) Deploy API on Render (public URL)

1. Push this repo to GitHub.
2. In Render, create a new **Web Service** from the repo.
3. Use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `PYTHONPATH=src uvicorn incentive_model.api:app --host 0.0.0.0 --port $PORT`
4. Upload your trained artifact to your service filesystem at:
   - `/opt/render/project/src/artifacts/incentive_recommender.joblib`
5. Set env var `MODEL_PATH=/opt/render/project/src/artifacts/incentive_recommender.joblib`.

Once deployed, your interactive endpoints will be:

- `https://<your-render-service>.onrender.com/docs`
- `https://<your-render-service>.onrender.com/recommend`

## Notes

- Feature engineering avoids target leakage by not deriving inputs from `ideal_incentive_program`.
- The trainer handles mixed numeric/categorical features automatically.
- Missing values are imputed during data cleaning.
- The baseline model uses a tuned `RandomForestClassifier` and can be replaced with XGBoost/LightGBM.
