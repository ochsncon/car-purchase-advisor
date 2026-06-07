"""Recommendation engine combining rules and optional LLM explanation."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.config import LLM_API_KEY, LLM_MODEL, LLM_PROMPT_VERSION
from src.utils import format_currency_chf


def _assess_price_vs_budget(estimated_price: float, budget: float) -> tuple[str, str]:
    if budget <= 0:
        return "Budget not provided", "Enter a budget to compare it with the estimated price."

    ratio = estimated_price / budget if budget else 0.0
    if ratio <= 0.9:
        return "Within budget", "The estimated price is clearly below the budget."
    if ratio <= 1.1:
        return "Close to budget", "The estimated price is near the entered budget."
    return "Above budget", "The estimated price is above the entered budget."


def _financing_orientation(
    budget: float,
    estimated_price: float,
    max_monthly_rate: float,
) -> dict[str, Any]:
    if budget >= estimated_price:
        return {
            "financing_orientation": "Realistic",
            "financing_gap": 0.0,
            "rough_months_needed": 0.0,
            "financing_reason": "Buying with own funds looks realistic. Keep a reserve for registration and maintenance.",
        }

    financing_gap = max(0.0, estimated_price - budget)
    if max_monthly_rate <= 0:
        return {
            "financing_orientation": "Unrealistic",
            "financing_gap": financing_gap,
            "rough_months_needed": None,
            "financing_reason": "No monthly rate was provided, so the financing gap cannot be translated into months.",
        }

    rough_months_needed = financing_gap / max_monthly_rate
    if rough_months_needed <= 12:
        orientation = "Realistic"
    elif rough_months_needed <= 24:
        orientation = "Tight"
    else:
        orientation = "Unrealistic"

    return {
        "financing_orientation": orientation,
        "financing_gap": financing_gap,
        "rough_months_needed": rough_months_needed,
        "financing_reason": (
            f"The financing gap is {format_currency_chf(financing_gap)}. "
            f"At {format_currency_chf(max_monthly_rate)} per month, this is about {rough_months_needed:.1f} months."
        ),
    }


def _build_llm_prompt_structured(
    user_inputs: dict[str, Any],
    vision_results: dict[str, Any],
    price_prediction: dict[str, Any],
    financing_text: str,
    budget_assessment: str,
    budget_reason: str,
) -> str:
    return f"""
You are an assistant for a used-car orientation app.
Write in concise, plain English for non-experts.

Rules:
- This is only a first orientation and not binding advice.
- Do not claim technical diagnosis from the image.
- Mention limitations clearly.

Structured inputs:
- Vision predicted class/model group: {vision_results.get('predicted_class')}
- Vision confidence: {vision_results.get('confidence')}
- Estimated price: {price_prediction.get('estimated_price')} CHF
- Estimated range: {price_prediction.get('lower_bound')} - {price_prediction.get('upper_bound')} CHF
- Budget: {user_inputs.get('budget_chf')} CHF
- Max monthly rate: {user_inputs.get('max_monthly_rate_chf')} CHF

Derived recommendations:
- Price vs budget: {budget_assessment}
- Budget reason: {budget_reason}
- Financing orientation: {financing_text}

Write one short paragraph only. Mention the predicted class, the price range, the budget assessment, the simple financing orientation, and the main limitations.
""".strip()


def _build_llm_prompt_concise(
    user_inputs: dict[str, Any],
    vision_results: dict[str, Any],
    price_prediction: dict[str, Any],
    financing_text: str,
    budget_assessment: str,
) -> str:
    return f"""
Short and clear in English. Orientation only, not binding advice.

Image: {vision_results.get('predicted_class')} ({vision_results.get('confidence')})
Price: {price_prediction.get('estimated_price')} CHF, range {price_prediction.get('lower_bound')} - {price_prediction.get('upper_bound')} CHF
Budget: {user_inputs.get('budget_chf')} CHF
Monthly rate: {user_inputs.get('max_monthly_rate_chf')} CHF

Assessment: {budget_assessment}
Financing orientation: {financing_text}

Reply with 1 compact paragraph. No premiums, no interest rates, no technical diagnosis.
""".strip()


def _build_llm_prompt(
    user_inputs: dict[str, Any],
    vision_results: dict[str, Any],
    price_prediction: dict[str, Any],
    financing_text: str,
    budget_assessment: str,
    budget_reason: str,
) -> str:
    if LLM_PROMPT_VERSION == "concise":
        return _build_llm_prompt_concise(
            user_inputs,
            vision_results,
            price_prediction,
            financing_text,
            budget_assessment,
        )
    return _build_llm_prompt_structured(
        user_inputs,
        vision_results,
        price_prediction,
        financing_text,
        budget_assessment,
        budget_reason,
    )


def _call_llm(prompt: str) -> str | None:
    if not LLM_API_KEY:
        return None

    try:
        client = OpenAI(api_key=LLM_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a cautious automotive purchase advisor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=450,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return None


def generate_recommendation(
    user_inputs: dict[str, Any],
    vision_results: dict[str, Any],
    price_prediction: dict[str, Any],
) -> dict[str, str]:
    """Generate budget assessment, financing orientation and explanation text."""
    estimated_price = float(price_prediction.get("estimated_price", 0) or 0)
    budget = float(user_inputs.get("budget_chf", 0) or 0)
    max_monthly_rate = float(user_inputs.get("max_monthly_rate_chf", 0) or 0)

    budget_assessment, budget_reason = _assess_price_vs_budget(estimated_price, budget)
    financing_payload = _financing_orientation(budget, estimated_price, max_monthly_rate)

    prompt = _build_llm_prompt(
        user_inputs=user_inputs,
        vision_results=vision_results,
        price_prediction=price_prediction,
        financing_text=financing_payload["financing_orientation"],
        budget_assessment=budget_assessment,
        budget_reason=budget_reason,
    )

    llm_text = _call_llm(prompt)
    if llm_text:
        explanation = llm_text.strip()
    else:
        explanation = (
            f"The image suggests '{vision_results.get('predicted_class', 'Unknown')}'. "
            f"The estimated price is about {price_prediction.get('estimated_price')} CHF "
            f"with a range of {price_prediction.get('lower_bound')} to {price_prediction.get('upper_bound')} CHF. "
            f"{budget_assessment}: {budget_reason} "
            f"{financing_payload['financing_reason']} "
            "This is a short orientation only and does not replace professional advice."
        )

    return {
        "price_budget_assessment": budget_assessment,
        "price_budget_reason": budget_reason,
        "financing_orientation": financing_payload["financing_orientation"],
        "financing_gap": financing_payload["financing_gap"],
        "rough_months_needed": financing_payload["rough_months_needed"],
        "financing_reason": financing_payload["financing_reason"],
        "full_explanation": explanation,
        "prompt_version": LLM_PROMPT_VERSION,
    }
