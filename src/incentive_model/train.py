from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from incentive_model.data_pipeline import create_training_dataset


def build_model(X: pd.DataFrame) -> Pipeline:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def train_and_evaluate(
    incentive_path: Path,
    property_path: Path,
    behavior_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
) -> None:
    incentive_df = pd.read_csv(incentive_path)
    property_df = pd.read_csv(property_path)
    behavior_df = pd.read_csv(behavior_path)

    dataset = create_training_dataset(incentive_df, property_df, behavior_df)
    df = dataset.training_df

    X = df[dataset.feature_columns]
    y = df[dataset.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    pipeline = build_model(X)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "incentive_recommender.joblib"
    metrics_path = output_dir / "metrics.json"

    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": dataset.feature_columns,
            "target_column": dataset.target_column,
        },
        model_path,
    )

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Evaluation accuracy: {accuracy:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train incentive recommendation model")
    parser.add_argument("--incentive-data", required=True, type=Path)
    parser.add_argument("--property-data", required=True, type=Path)
    parser.add_argument("--behavior-data", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("artifacts"), type=Path)
    parser.add_argument("--test-size", default=0.2, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(
        incentive_path=args.incentive_data,
        property_path=args.property_data,
        behavior_path=args.behavior_data,
        output_dir=args.output_dir,
        test_size=args.test_size,
    )
