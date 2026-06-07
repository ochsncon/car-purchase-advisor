"""Central configuration for paths and environment settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"

PRICE_MODEL_PATH = MODEL_DIR / "price_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
LLM_PROMPT_VERSION = os.getenv("LLM_PROMPT_VERSION", "structured").strip().lower()

MAX_TRAINING_ROWS = int(os.getenv("MAX_TRAINING_ROWS", "30000"))

# Optional files for local CV model inference.
VISION_MODEL_PATH = MODEL_DIR / "vision_model.pkl"
VISION_LABELS_PATH = MODEL_DIR / "vision_labels.json"
VISION_METADATA_PATH = MODEL_DIR / "vision_metadata.json"

RAW_DATA_CANDIDATES = [
    "CarsDatasets2025.csv",
    "car_listings.csv",
    "autoscout24.csv",
    "vehicles.csv",
    "cars.csv",
]
