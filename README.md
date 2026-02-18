# Incentive Recommendation ML Pipeline

This project provides an end-to-end Python workflow for predicting an **ideal incentive program for property owners** using:

- Raw Incentive Data
- Raw Property Data
- User Behavior Data

It follows this pipeline:

```text
Raw Incentive Data
Raw Property Data
User Behavior Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Training Dataset Creation
        ↓
Model Training
        ↓
Evaluation
        ↓
Model Deployment (API)
        ↓
Real-Time Incentive Recommendations
```

## Project Structure

```text
src/incentive_model/
  data_pipeline.py   # cleaning + feature engineering + training set creation
  train.py           # model training + evaluation + artifact creation
  api.py             # FastAPI service for real-time predictions
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

## 4) Deploy API

```bash
PYTHONPATH=src uvicorn incentive_model.api:app --host 0.0.0.0 --port 8000
```

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
      "avg_program_incentive_amount": 1800,
      "owner_property_interaction": "O-102_P-21"
    }
  }'
```

The API response contains the recommended incentive program and class probabilities.

## Notes

- The trainer automatically handles mixed numeric/categorical features.
- Missing values are imputed during data cleaning.
- Categorical values are one-hot encoded.
- The baseline model uses a `RandomForestClassifier` and can be replaced with XGBoost/LightGBM if needed.
