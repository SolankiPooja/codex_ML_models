from __future__ import annotations

from dataclasses import dataclass

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


def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median() if not df[col].dropna().empty else 0
            df[col] = df[col].fillna(fill_val)
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")
    return df


def clean_data(
    incentive_df: pd.DataFrame,
    property_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Basic cleaning and schema checks for all raw datasets."""
    _validate_columns(incentive_df, REQUIRED_INCENTIVE_COLUMNS, "Raw incentive data")
    _validate_columns(property_df, REQUIRED_PROPERTY_COLUMNS, "Raw property data")
    _validate_columns(behavior_df, REQUIRED_BEHAVIOR_COLUMNS, "User behavior data")

    incentive_df = _fill_missing(_normalize_strings(incentive_df.copy().drop_duplicates()))
    property_df = _fill_missing(_normalize_strings(property_df.copy().drop_duplicates()))
    behavior_df = _fill_missing(_normalize_strings(behavior_df.copy().drop_duplicates()))
    return incentive_df, property_df, behavior_df


def feature_engineering(
    incentive_df: pd.DataFrame,
    property_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
) -> PipelineData:
    """Build training features without leaking target label information."""
    merged = behavior_df.merge(property_df, on=["owner_id", "property_id"], how="left")

    # Global incentive context features (available at both train and inference time)
    merged["global_avg_incentive_amount"] = float(incentive_df["incentive_amount"].mean())
    merged["global_max_incentive_amount"] = float(incentive_df["incentive_amount"].max())
    merged["global_min_incentive_amount"] = float(incentive_df["incentive_amount"].min())
    merged["available_program_count"] = int(incentive_df["incentive_program"].nunique())

    # Owner-level historical behavior aggregates
    owner_stats = (
        behavior_df.groupby("owner_id", as_index=False)["engagement_score"]
        .agg(owner_avg_engagement="mean", owner_max_engagement="max", owner_interaction_count="count")
    )
    merged = merged.merge(owner_stats, on="owner_id", how="left")

    # Interaction key used by inference fallback logic
    merged["owner_property_interaction"] = merged["owner_id"].astype(str) + "_" + merged["property_id"].astype(str)

    training_df = _fill_missing(merged)
    feature_columns = [c for c in training_df.columns if c != "ideal_incentive_program"]
    return PipelineData(training_df=training_df, feature_columns=feature_columns)


def create_training_dataset(
    incentive_df: pd.DataFrame,
    property_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
) -> PipelineData:
    clean_incentive, clean_property, clean_behavior = clean_data(incentive_df, property_df, behavior_df)
    return feature_engineering(clean_incentive, clean_property, clean_behavior)
