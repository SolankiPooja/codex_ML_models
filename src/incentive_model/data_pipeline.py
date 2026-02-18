from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


REQUIRED_INCENTIVE_COLUMNS = {"incentive_program", "incentive_amount"}
REQUIRED_PROPERTY_COLUMNS = {"property_id", "owner_id"}
REQUIRED_BEHAVIOR_COLUMNS = {"owner_id", "property_id", "engagement_score", "ideal_incentive_program"}


@dataclass
class PipelineData:
    training_df: pd.DataFrame
    feature_columns: list[str]
    target_column: str = "ideal_incentive_program"


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def clean_data(
    incentive_df: pd.DataFrame,
    property_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Basic cleaning for all raw datasets."""
    _validate_columns(incentive_df, REQUIRED_INCENTIVE_COLUMNS, "Raw incentive data")
    _validate_columns(property_df, REQUIRED_PROPERTY_COLUMNS, "Raw property data")
    _validate_columns(behavior_df, REQUIRED_BEHAVIOR_COLUMNS, "User behavior data")

    incentive_df = incentive_df.copy()
    property_df = property_df.copy()
    behavior_df = behavior_df.copy()

    for df in (incentive_df, property_df, behavior_df):
        df.drop_duplicates(inplace=True)

    # Standardize string columns
    for col in incentive_df.select_dtypes(include="object").columns:
        incentive_df[col] = incentive_df[col].str.strip()

    for col in property_df.select_dtypes(include="object").columns:
        property_df[col] = property_df[col].str.strip()

    for col in behavior_df.select_dtypes(include="object").columns:
        behavior_df[col] = behavior_df[col].str.strip()

    # Fill null numerics with median and categoricals with mode
    def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")
        return df

    return fill_missing(incentive_df), fill_missing(property_df), fill_missing(behavior_df)


def feature_engineering(
    incentive_df: pd.DataFrame,
    property_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
) -> PipelineData:
    """Merge and engineer features for model training."""
    incentives_by_program = (
        incentive_df.groupby("incentive_program", as_index=False)["incentive_amount"]
        .mean()
        .rename(columns={"incentive_amount": "avg_program_incentive_amount"})
    )

    merged = behavior_df.merge(property_df, on=["owner_id", "property_id"], how="left")
    merged = merged.merge(
        incentives_by_program,
        left_on="ideal_incentive_program",
        right_on="incentive_program",
        how="left",
    )

    merged["avg_program_incentive_amount"] = merged["avg_program_incentive_amount"].fillna(
        incentives_by_program["avg_program_incentive_amount"].median()
    )

    merged["owner_property_interaction"] = merged["owner_id"].astype(str) + "_" + merged["property_id"].astype(str)

    drop_cols = ["incentive_program"] if "incentive_program" in merged.columns else []
    training_df = merged.drop(columns=drop_cols)

    feature_columns = [c for c in training_df.columns if c != "ideal_incentive_program"]
    return PipelineData(training_df=training_df, feature_columns=feature_columns)


def create_training_dataset(
    incentive_df: pd.DataFrame,
    property_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
) -> PipelineData:
    clean_incentive, clean_property, clean_behavior = clean_data(incentive_df, property_df, behavior_df)
    return feature_engineering(clean_incentive, clean_property, clean_behavior)
