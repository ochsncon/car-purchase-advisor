"""Gradio frontend for AI Car Purchase Advisor."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

from src.price_predictor import predict_price
from src import price_predictor
from src.recommendation_engine import generate_recommendation
from src.utils import format_currency_chf, to_float
from src.vision_analyzer import analyze_car_image


def _normalize_age(year_or_age_value: float | int | None, mode: str) -> float:
    value = to_float(year_or_age_value, 0)
    if value <= 0:
        return 6.0

    if str(mode).strip().lower() == "year":
        current_year = datetime.now().year
        age = current_year - value
        if age < 0:
            return 0.0
        return float(age)

    return float(value)


def _build_vehicle_recognition_markdown(vision_results: dict[str, Any]) -> str:
    predicted_class = vision_results.get("predicted_class", "Unknown")
    confidence = float(vision_results.get("confidence", 0.0) or 0.0)
    method = vision_results.get("method", "fallback")

    # Case A: Image analyzed by vision model
    if method == "local_transfer_model":
        confidence_warning = ""
        if confidence < 0.5:
            confidence_warning = "\n\nThe image model is uncertain. Please verify the result manually."

        return f"""
### 1. Vehicle recognition – Computer Vision

Detected brand/model group: **{predicted_class}**

The image analysis provides a coarse vehicle brand/model prediction based on deep learning. It is not exact technical vehicle identification.{confidence_warning}
""".strip()

    # Case B: Manual input without image
    elif method == "manual_input":
        return f"""
### 1. Vehicle recognition – Manual Input

Vehicle used for the estimate: **{predicted_class}**

The price estimate is based on your manual vehicle input and structured used-car listing data.
""".strip()

    # Case C: Unknown or fallback
    else:
        return f"""
### 1. Vehicle recognition

Vehicle: **Unknown**

Please upload a car image or enter a known make/model to get a price estimate.
""".strip()


def _build_price_estimate_markdown(price_prediction: dict[str, Any]) -> str:
    estimated_price = format_currency_chf(price_prediction.get("estimated_price"))
    lower_bound = format_currency_chf(price_prediction.get("lower_bound"))
    upper_bound = format_currency_chf(price_prediction.get("upper_bound"))

    return f"""
### 2. Price estimate – ML Numeric Data

- Estimated market price: **{estimated_price}**
- Expected price range: **{lower_bound} – {upper_bound}**

This estimate is based on structured used-car listing data.
""".strip()


def _build_purchase_assessment_markdown(recommendation: dict[str, Any]) -> str:
    rough_months_needed = recommendation.get("rough_months_needed")
    financing_gap = recommendation.get("financing_gap")

    months_text = "n/a" if rough_months_needed in {None, "", 0} else f"{float(rough_months_needed):.1f} months"
    gap_text = "n/a" if financing_gap in {None, ""} else format_currency_chf(financing_gap)

    budget_assessment = recommendation.get('price_budget_assessment', 'n/a')
    budget_reason = recommendation.get('price_budget_reason', '')
    financing_orientation = recommendation.get('financing_orientation', 'n/a')
    financing_reason = recommendation.get('financing_reason', '')
    explanation = recommendation.get('full_explanation', 'n/a')

    return f"""
### 3. Purchase assessment – NLP Explanation

**Budget assessment:** {budget_assessment}
{budget_reason}

**Simple financing orientation:** {financing_orientation}
- Financing gap: {gap_text}
- Rough months needed: {months_text}
{financing_reason}

**Assessment:**
{explanation}
""".strip()


def _build_debug_payload(
    vision_results: dict[str, Any],
    price_prediction: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "vision": vision_results,
        "price": price_prediction,
        "recommendation": recommendation,
    }


def _collect_example_images(limit: int = 3) -> list[list[str]]:
    project_root = Path(__file__).resolve().parent
    examples_dir = project_root / "example_images"

    allowed_ext = {".jpg", ".jpeg", ".png", ".webp"}

    if not examples_dir.exists():
        print(f"Example image folder not found: {examples_dir}")
        return []

    image_paths = sorted(
        [
            path
            for path in examples_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in allowed_ext
        ]
    )

    examples = [[str(path)] for path in image_paths[:limit]]

    print("Loaded example images:", examples)
    return examples


def _run_advisor(
    image,
    make_model,
    mileage_km,
    car_age_years,
    budget_chf,
    max_monthly_rate_chf,
):
    # Allow either image OR manual make_model input, but require at least one
    if image is None and not (make_model and make_model.strip()):
        return (
            "Please upload a car image OR enter a known make/model to start the analysis.",
            "",
            "",
            {},
        )

    # If image is provided, use vision analyzer; otherwise use manual input
    if image is not None:
        vision_results = analyze_car_image(image)
        predicted_class = vision_results.get("predicted_class", "Unknown")
    else:
        # No image, but make_model was entered manually
        predicted_class = make_model.strip() if make_model and make_model.strip() else "Unknown"
        vision_results = {
            "predicted_class": predicted_class,
            "confidence": 0.0,
            "method": "manual_input",
            "notes": ["Car brand/model entered manually without image."],
        }

    age = _normalize_age(car_age_years, "Age")
    make_model_input = make_model.strip() if make_model and make_model.strip() else predicted_class

    # For manual input without image, use heuristic (ML model needs too many features we don't have)
    # For image-based input, use ML model if available
    if vision_results.get("method") == "manual_input":
        # Manual input: use heuristic with make_model for brand detection
        price_input = {
            "make_model": make_model_input,
            "hp_kW": None,
            "Fuel": None,
        }
        price_prediction = price_predictor._heuristic_estimate(price_input)
    else:
        # Image-based: try ML model with available features
        price_input = {
            "make_model": make_model_input,
            "hp_kW": None,
            "Fuel": None,
        }
        price_prediction = predict_price(price_input)

    user_inputs = {
        "budget_chf": to_float(budget_chf, 0),
        "max_monthly_rate_chf": to_float(max_monthly_rate_chf, 0),
        "car_age": age,
        "km": to_float(mileage_km, 0),
    }

    recommendation = generate_recommendation(
        user_inputs=user_inputs,
        vision_results=vision_results,
        price_prediction=price_prediction,
    )

    return (
        _build_vehicle_recognition_markdown(vision_results),
        _build_price_estimate_markdown(price_prediction),
        _build_purchase_assessment_markdown(recommendation),
        _build_debug_payload(vision_results, price_prediction, recommendation),
    )


def _clear_outputs():
    return "", "", "", {}


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="AI Car Purchase Advisor") as demo:
        gr.Markdown(
            """
# AI Car Purchase Advisor

Upload a car image to estimate the vehicle class, market price range and receive a short AI-generated purchase assessment.

Image upload -> analyze_car_image() -> predicted_class -> predict_price() -> estimated_price/range -> generate_recommendation() -> final explanation
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload car image")
                make_model = gr.Textbox(label="Known make/model", placeholder="e.g. Audi A3")
                mileage_km = gr.Number(label="Mileage in km", value=80000)
                car_age_years = gr.Number(label="Car age in years", value=6)
                budget_chf = gr.Number(label="Budget in CHF", value=25000)
                max_monthly_rate_chf = gr.Number(label="Maximum monthly rate in CHF", value=450)

                with gr.Row():
                    submit_btn = gr.Button("Generate assessment", variant="primary")
                    clear_btn = gr.Button("Clear")

                with gr.Accordion("Example images", open=False):
                    examples = _collect_example_images()
                    if examples:
                        gr.Examples(examples=examples, inputs=image_input, outputs=None, label="Click an example to load it")
                    else:
                        gr.Markdown("No example images found in the training dataset.")

            with gr.Column(scale=1):
                vehicle_recognition = gr.Markdown(label="Vehicle recognition")
                price_estimate = gr.Markdown(label="Price estimate")
                purchase_assessment = gr.Markdown(label="Purchase assessment")

                with gr.Accordion("Debug details", open=False):
                    debug_output = gr.JSON(label="Raw pipeline output")

        submit_btn.click(
            fn=_run_advisor,
            inputs=[image_input, make_model, mileage_km, car_age_years, budget_chf, max_monthly_rate_chf],
            outputs=[
                vehicle_recognition,
                price_estimate,
                purchase_assessment,
                debug_output,
            ],
        )

        clear_btn.click(
            fn=_clear_outputs,
            inputs=[],
            outputs=[
                vehicle_recognition,
                price_estimate,
                purchase_assessment,
                debug_output,
            ],
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch(show_error=True)


if __name__ == "__main__":
    main()
