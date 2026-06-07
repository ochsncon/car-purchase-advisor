"""Computer Vision analyzer using transfer-learning ResNet-18 for vehicle brand classification.

The module loads a Hugging Face fine-tuned image classification model from
models/car-image-classifier/ and provides vehicle brand predictions.

No damage detection and no technical condition estimation is implemented.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config import MODEL_DIR


def ensure_pil_image(image: Image.Image | np.ndarray | None) -> Image.Image:
    """Convert input to PIL Image."""
    if image is None:
        raise ValueError("No image provided.")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype("uint8"))
    if not isinstance(image, Image.Image):
        raise TypeError("Input is not a valid image.")
    return image


@lru_cache(maxsize=1)
def _load_transfer_model() -> dict[str, Any] | None:
    """Load the transfer-learning model from models/car-image-classifier/ or from HF Hub.
    
    Returns a dict with 'pipeline' and 'metadata' on success, None if model not found.
    """
    from transformers import pipeline

    model_dir = MODEL_DIR / "car-image-classifier"
    model_source = None
    metadata = {}

    # Try to load locally first
    if model_dir.exists():
        model_source = str(model_dir)
        metadata_path = model_dir / "vision_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        # Fall back to Hugging Face Hub
        model_source = "ochsncon/car-image-classifier"

    try:
        clf_pipeline = pipeline("image-classification", model=model_source, device=-1)
        return {"pipeline": clf_pipeline, "metadata": metadata}
    except Exception:
        return None


def analyze_car_image(image: Image.Image | np.ndarray | None) -> dict[str, Any]:
    """Analyze uploaded image and return vehicle brand prediction.
    
    Uses a transfer-learning ResNet-18 model to classify vehicle brands.
    """
    if image is None:
        return {
            "predicted_class": "Unknown",
            "confidence": 0.0,
            "method": "no_image",
            "notes": ["No image was provided."],
        }

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))

    if not isinstance(image, Image.Image):
        return {
            "predicted_class": "Unknown",
            "confidence": 0.0,
            "method": "invalid_input",
            "notes": ["Input is not a valid image format."],
        }

    model_bundle = _load_transfer_model()
    if model_bundle is not None:
        try:
            pipeline = model_bundle["pipeline"]
            metadata = model_bundle.get("metadata", {})

            # Run inference
            pil_image = ensure_pil_image(image)
            results = pipeline(pil_image, top_k=1)

            if results:
                top_result = results[0]
                predicted = top_result["label"]
                confidence = float(top_result["score"])

                return {
                    "predicted_class": predicted,
                    "confidence": round(confidence, 3),
                    "method": "local_transfer_model",
                    "notes": [
                        "Vehicle brand classification using transfer learning (ResNet-18).",
                        "The classifier can only predict one of the trained classes.",
                        f"Model accuracy on test set: {metadata.get('accuracy', 'n/a')}.",
                        "No damage detection or technical condition assessment.",
                    ],
                }
        except Exception:
            pass

    # Fallback if model not found or inference fails
    return {
        "predicted_class": "Unknown",
        "confidence": 0.0,
        "method": "fallback",
        "notes": [
            "No trained transfer-learning model found.",
            "Please train the model using: python -m src.train_vision_model",
        ],
    }
