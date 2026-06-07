"""Data loading, cleaning and feature engineering for used-car price estimation.

This module provides helpers to load a CSV from data/raw, normalize common
external column name variants, and build the feature frames used by the price
estimator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import io
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_RAW_DIR, MAX_TRAINING_ROWS, RAW_DATA_CANDIDATES
from src.utils import bool_from_yes_no, compute_age_from_year, count_equipment_items, to_float

TARGET_COLUMN = "price"

# Simplified features - only those available in CarsDatasets2025.csv
BASE_FEATURES = [
    "make_model",
    "hp_kW",
    "Fuel",
]

EQUIPMENT_COLUMNS = []


def _find_first_csv_in_raw() -> Path | None:
    for candidate in RAW_DATA_CANDIDATES:
        path = DATA_RAW_DIR / candidate
        if path.exists() and path.is_file():
            return path

    csv_files = sorted(DATA_RAW_DIR.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    return None


def load_car_data(max_rows: int | None = None) -> tuple[pd.DataFrame, Path]:
    """Load a CSV from data/raw and normalize common external column variants.

    This function requires a real CSV in data/raw; it will raise a
    FileNotFoundError with guidance if none is found. It tries several
    encodings to avoid UnicodeDecodeError on files from different sources.
    """
    data_path = _find_first_csv_in_raw()
    if data_path is None:
        raise FileNotFoundError(
            f"No raw CSV found in {DATA_RAW_DIR}. Place your dataset CSV in this folder. "
            f"Searched candidates: {RAW_DATA_CANDIDATES} and top-level .csv files."
        )

    def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            encodings = ["latin-1", "cp1252", "ISO-8859-1"]
            for enc in encodings:
                try:
                    df_try = pd.read_csv(path, encoding=enc, low_memory=False)
                    print(f"Loaded CSV using encoding: {enc}")
                    return df_try
                except Exception:
                    continue

            with open(path, "rb") as fh:
                raw = fh.read()
            text = raw.decode("utf-8", errors="replace")
            return pd.read_csv(io.StringIO(text), low_memory=False)

    df = _read_csv_with_fallback(data_path)

    # Normalize column names by stripping whitespace (some CSV exports include
    # trailing spaces in headers, e.g. 'Cars Prices '). This makes mapping more
    # robust.
    df.columns = [str(c).strip() for c in df.columns]

    # Helper parsers for messy external columns
    def _parse_price(val: Any) -> float | None:
        if pd.isna(val):
            return None
        s = str(val)
        s = s.replace("$", "").replace("€", "").replace("CHF", "")
        s = s.replace("–", "-").replace("—", "-")
        nums = re.findall(r"[0-9]+(?:[\.,][0-9]+)?", s)
        if not nums:
            return None
        vals = []
        for n in nums:
            n_clean = n.replace(",", "").replace(" ", "")
            try:
                vals.append(float(n_clean))
            except Exception:
                continue
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def _parse_cc(val: Any) -> float | None:
        if pd.isna(val):
            return None
        s = str(val)
        m = re.search(r"([0-9]+(?:[\.,][0-9]+)?)\s*cc", s, flags=re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", ""))
        m = re.search(r"([0-9]+(?:[\.,][0-9]+)?)\s*L", s, flags=re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", "")) * 1000.0
        nums = re.findall(r"[0-9]+(?:[\.,][0-9]+)?", s)
        if nums:
            return float(nums[0].replace(",", ""))
        return None

    def _parse_hp(val: Any) -> float | None:
        if pd.isna(val):
            return None
        s = str(val)
        nums = re.findall(r"[0-9]+(?:[\.,][0-9]+)?", s)
        if not nums:
            return None
        vals = [float(n.replace(",", "")) for n in nums]
        return float(sum(vals) / len(vals))

    # Map common external columns to our expected names
    col_map = {}
    if "Cars Prices" in df.columns:
        col_map["Cars Prices"] = "price"
    if "Cars Names" in df.columns and "Company Names" in df.columns:
        df["make_model"] = (df["Company Names"].fillna("") + " " + df["Cars Names"].fillna(""))
        df["make_model"] = df["make_model"].str.strip()
    elif "Cars Names" in df.columns:
        df["make_model"] = df["Cars Names"].astype(str)
    if "Fuel Types" in df.columns:
        col_map["Fuel Types"] = "Fuel"
    if "CC/Battery Capacity" in df.columns:
        col_map["CC/Battery Capacity"] = "Displacement_cc"
    if "HorsePower" in df.columns:
        col_map["HorsePower"] = "hp_kW"

    if col_map:
        df = df.rename(columns=col_map)

    if "price" in df.columns:
        df["price"] = df["price"].apply(_parse_price)
    if "Displacement_cc" in df.columns:
        df["Displacement_cc"] = df["Displacement_cc"].apply(_parse_cc)
    if "hp_kW" in df.columns:
        df["hp_kW"] = df["hp_kW"].apply(_parse_hp)
        df["hp_kW"] = df["hp_kW"].apply(lambda v: v * 0.7457 if pd.notna(v) and v > 50 else v)

    if "Inspection_new" in df.columns:
        df["Inspection_new"] = df["Inspection_new"].apply(bool_from_yes_no)

    # Optionally limit rows (we do sampling downstream in prepare_train_test_data)
    nrows = max_rows if max_rows is not None else -1
    if nrows > 0 and len(df) > nrows:
        df = df.sample(n=nrows, random_state=42).reset_index(drop=True)

    return df, data_path


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BASE_FEATURES + EQUIPMENT_COLUMNS + [TARGET_COLUMN, "year"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Only hp_kW is available in CarsDatasets2025.csv
    numeric_columns = [
        "price",
        "hp_kW",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(to_float)

    return df


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Only available features in CarsDatasets2025.csv
    if "hp_kW" in df.columns:
        df["hp_kW"] = df["hp_kW"].clip(lower=20, upper=800)

    return df


def clean_and_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_required_columns(df)
    df = _clean_numeric_columns(df)
    df = _feature_engineering(df)

    df = df.dropna(subset=[TARGET_COLUMN])

    # Remove unrealistic price entries and outliers.
    df = df[(df[TARGET_COLUMN] > 1000) & (df[TARGET_COLUMN] < 500000)]
    if len(df) > 20:
        lower = df[TARGET_COLUMN].quantile(0.01)
        upper = df[TARGET_COLUMN].quantile(0.99)
        df = df[(df[TARGET_COLUMN] >= lower) & (df[TARGET_COLUMN] <= upper)]

    return df


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return model feature frame with engineered columns."""
    feature_columns = BASE_FEATURES
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_columns].copy()


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = [
        "hp_kW",
    ]

    categorical_features = [
        "make_model",
        "Fuel",
    ]

    for col in numeric_features + categorical_features:
        if col not in X.columns:
            X[col] = np.nan

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def prepare_train_test_data(test_size: float = 0.2, random_state: int = 42) -> dict[str, Any]:
    df_raw, data_path = load_car_data()
    df_clean = clean_and_prepare_dataframe(df_raw)

    X = build_feature_dataframe(df_clean)
    y = df_clean[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

    return {
        "raw_df": df_raw,
        "clean_df": df_clean,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "data_path": data_path,
    }


def prepare_inference_input(input_data: dict[str, Any]) -> pd.DataFrame:
    """Prepare one-row inference DataFrame from user input."""
    row = {
        "make_model": input_data.get("make_model"),
        "body_type": input_data.get("body_type"),
        "km": input_data.get("km"),
        "age": input_data.get("age"),
        "hp_kW": input_data.get("hp_kW"),
        "Fuel": input_data.get("Fuel"),
        "Gearing_Type": input_data.get("Gearing_Type"),
        "Displacement_cc": input_data.get("Displacement_cc"),
        "Weight_kg": input_data.get("Weight_kg"),
        "Drive_chain": input_data.get("Drive_chain"),
        "cons_comb": input_data.get("cons_comb"),
        "Previous_Owners": input_data.get("Previous_Owners"),
        "Inspection_new": input_data.get("Inspection_new"),
        "Comfort_Convenience": input_data.get("Comfort_Convenience", ""),
        "Entertainment_Media": input_data.get("Entertainment_Media", ""),
        "Extras": input_data.get("Extras", ""),
        "Safety_Security": input_data.get("Safety_Security", ""),
    }

    df = pd.DataFrame([row])
    df = _ensure_required_columns(df)
    df = _clean_numeric_columns(df)
    df = _feature_engineering(df)
    df_features = build_feature_dataframe(df)
    return df_features
