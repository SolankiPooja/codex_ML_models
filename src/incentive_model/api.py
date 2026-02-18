from __future__ import annotations

import os
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/incentive_recommender.joblib")


class RecommendationRequest(BaseModel):
    features: dict[str, Any]


app = FastAPI(title="Incentive Recommendation API", version="1.0.0")


@app.on_event("startup")
def load_model() -> None:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Train model first.")

    bundle = joblib.load(MODEL_PATH)
    app.state.pipeline = bundle["pipeline"]
    app.state.feature_columns = bundle["feature_columns"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecommendationRequest) -> dict[str, Any]:
    feature_columns = app.state.feature_columns

    missing_features = [col for col in feature_columns if col not in req.features]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing_features}",
        )

    input_df = pd.DataFrame([req.features])[feature_columns]
    pipeline = app.state.pipeline

    prediction = pipeline.predict(input_df)[0]

    response: dict[str, Any] = {"recommended_incentive_program": prediction}

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_df)[0]
        labels = pipeline.classes_
        response["class_probabilities"] = {
            label: float(prob) for label, prob in zip(labels, probs)
        }

    return response
