"""Deterministic MGSM answer and AXBench Spanish-rule scorers."""

from __future__ import annotations

import re

import langdetect


NUMBER_PATTERN = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def extract_final_number(completion: str) -> int | float | None:
    """Extract the final Arabic number independently of answer-prefix language."""

    matches = NUMBER_PATTERN.findall(completion.replace("\u2212", "-"))
    if not matches:
        return None
    normalized = matches[-1].replace(",", "")
    try:
        value = float(normalized)
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def exact_match(completion: str, answer_number: int | float) -> dict[str, object]:
    """Score the extracted final number against MGSM's numeric gold answer."""

    prediction = extract_final_number(completion)
    return {
        "prediction": prediction,
        "answer_number": answer_number,
        "score": float(prediction is not None and prediction == answer_number),
        "valid": prediction is not None,
    }


def spanish_rule_score(completion: str) -> dict[str, object]:
    """Apply AXBench's deterministic Spanish-only language rule."""

    text = completion.replace("<end_of_turn>", "").strip()
    try:
        detected = langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        detected = None
    return {
        "detected_language": detected,
        "score": 2.0 if detected == "es" else 0.0,
    }
