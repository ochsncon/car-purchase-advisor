"""Train and persist the used-car price model."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running via: python src/train_price_model.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODEL_DIR, MODEL_METADATA_PATH, PREPROCESSOR_PATH, PRICE_MODEL_PATH
from src.data_preprocessing import prepare_train_test_data


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    prepared = prepare_train_test_data()
    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"].values
    y_test = prepared["y_test"].values
    preprocessor = prepared["preprocessor"]

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    model_candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    }

    metrics = {}
    fitted_models = {}

    for name, model in model_candidates.items():
        model.fit(X_train_transformed, y_train)
        preds = model.predict(X_test_transformed)
        metrics[name] = evaluate_regression(y_test, preds)
        fitted_models[name] = model

    best_model_name = min(metrics.keys(), key=lambda m: metrics[m]["rmse"])
    best_model = fitted_models[best_model_name]

    joblib.dump(best_model, PRICE_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source": str(prepared["data_path"]),
        "model_name": best_model_name,
        "metrics": metrics,
        "best_metrics": metrics[best_model_name],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "training_columns": list(X_train.columns),
        "numeric_features": prepared["numeric_features"],
        "categorical_features": prepared["categorical_features"],
    }

    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Training completed.")
    print(f"Best model: {best_model_name}")
    print(f"Saved model to: {PRICE_MODEL_PATH}")
    print(f"Saved preprocessor to: {PREPROCESSOR_PATH}")
    print(f"Saved metadata to: {MODEL_METADATA_PATH}")

    for name, result in metrics.items():
        print(
            f"{name}: MAE={result['mae']:.2f}, RMSE={result['rmse']:.2f}, R2={result['r2']:.4f}"
        )


if __name__ == "__main__":
    main()
