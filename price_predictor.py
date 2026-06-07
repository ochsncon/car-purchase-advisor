"""Inference helper for used-car price estimation."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np

from src.config import MODEL_METADATA_PATH, PREPROCESSOR_PATH, PRICE_MODEL_PATH
from src.data_preprocessing import prepare_inference_input
from src.utils import to_float


def _heuristic_estimate(input_data: dict[str, Any]) -> dict[str, Any]:
    base = 26000.0
    age = to_float(input_data.get("age"), 6)
    km = to_float(input_data.get("km"), 90000)
    hp_kw = to_float(input_data.get("hp_kW"), 100)

    base -= age * 1500
    base -= (km / 1000) * 35
    base += (hp_kw - 100) * 45

    # Extract brand from make_model for adjustments
    make_model = str(input_data.get("make_model", "")).lower()
    luxury_adjustments = {
        "rolls": 70000,
        "bentley": 45000,
        "aston": 35000,
        "porsche": 30000,
        "mercedes": 12000,
        "bmw": 10000,
        "audi": 9000,
        "jaguar": 9000,
        "cadillac": 7000,
    }
    budget_adjustments = {
        "swift": -4000,
        "fiat": -2500,
        "hyundai": -1000,
        "kia": -1000,
        "mahindra": -500,
        "toyota": 2500,
        "mazda": 1500,
        "ford": 500,
    }

    applied = False
    for key, delta in luxury_adjustments.items():
        if key in make_model:
            base += delta
            applied = True
            break

    if not applied:
        for key, delta in budget_adjustments.items():
            if key in make_model:
                base += delta
                applied = True
                break

    if not applied and ("suv" in make_model or "benz" in make_model):
        base += 3500

    estimated = max(5000, min(180000, base))
    spread = max(2500, estimated * 0.12)

    return {
        "estimated_price": round(float(estimated), 2),
        "lower_bound": round(float(estimated - spread), 2),
        "upper_bound": round(float(estimated + spread), 2),
        "model_name": "HeuristicFallback",
        "confidence_note": "Estimated range from a simple age-and-mileage fallback because trained model artifacts are unavailable.",
    }


@lru_cache(maxsize=1)
def _load_model_bundle() -> dict[str, Any] | None:
    try:
        model = joblib.load(PRICE_MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        metadata = {}
        if MODEL_METADATA_PATH.exists():
            metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))

        return {
            "model": model,
            "preprocessor": preprocessor,
            "metadata": metadata,
        }
    except Exception:
        return None


def predict_price(input_data: dict[str, Any]) -> dict[str, Any]:
    """Predict used-car price and return estimate range."""
    model_bundle = _load_model_bundle()
    if model_bundle is None:
        return _heuristic_estimate(input_data)

    try:
        df_features = prepare_inference_input(input_data)
        X = model_bundle["preprocessor"].transform(df_features)
        prediction = float(model_bundle["model"].predict(X)[0])

        metadata = model_bundle.get("metadata", {})
        model_name = metadata.get("model_name", model_bundle["model"].__class__.__name__)
        rmse = (
            metadata.get("best_metrics", {}).get("rmse")
            if isinstance(metadata.get("best_metrics"), dict)
            else None
        )

        if rmse is None:
            spread = max(3000.0, prediction * 0.12)
            confidence_note = "Estimated range based on generic uncertainty band."
        else:
            spread = max(float(rmse), prediction * 0.08)
            confidence_note = "Estimated range based on validation RMSE of the trained model."

        lower = max(1000.0, prediction - spread)
        upper = prediction + spread

        return {
            "estimated_price": round(prediction, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "model_name": model_name,
            "confidence_note": confidence_note,
        }
    except Exception:
        return _heuristic_estimate(input_data)
