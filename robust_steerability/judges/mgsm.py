"""Deterministic MGSM answer and AXBench Spanish-rule scorers."""

from __future__ import annotations

import re


NUMBER_PATTERN = re.compile(r"-?\d+(?:[.,]\d+)*")


def _numeric_value(text: str) -> int | float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def _number_candidates(token: str) -> list[int | float]:
    """Return locale-aware interpretations of one formatted number token."""

    sign = ""
    unsigned = token
    if unsigned.startswith("-"):
        sign = "-"
        unsigned = unsigned[1:]
    separators = {separator for separator in ".," if separator in unsigned}
    normalized: list[str] = []
    if not separators:
        normalized.append(sign + unsigned)
    elif len(separators) == 2:
        decimal_separator = max(
            (unsigned.rfind("."), "."), (unsigned.rfind(","), ",")
        )[1]
        thousands_separator = "," if decimal_separator == "." else "."
        normalized.append(
            sign
            + unsigned.replace(thousands_separator, "").replace(
                decimal_separator, "."
            )
        )
    else:
        separator = next(iter(separators))
        groups = unsigned.split(separator)
        grouped_thousands = (
            len(groups) > 1
            and groups[0] != ""
            and all(len(group) == 3 for group in groups[1:])
        )
        if len(groups) > 2:
            if grouped_thousands:
                normalized.append(sign + "".join(groups))
        else:
            # A single three-digit suffix is ambiguous across locales: `8.000`
            # can mean eight thousand in Spanish while `8.000` can also mean
            # eight as a decimal. Retain both readings so exact matching can
            # resolve the token against the benchmark's numeric gold answer.
            decimal = sign + unsigned.replace(separator, ".")
            if grouped_thousands:
                normalized.append(sign + "".join(groups))
            normalized.append(decimal)

    candidates: list[int | float] = []
    for value_text in normalized:
        value = _numeric_value(value_text)
        if value is not None and value not in candidates:
            candidates.append(value)
    return candidates


def _final_number_candidates(completion: str) -> list[int | float]:
    matches = NUMBER_PATTERN.findall(completion.replace("\u2212", "-"))
    return _number_candidates(matches[-1]) if matches else []


def extract_final_number(completion: str) -> int | float | None:
    """Extract the final Arabic number independently of answer-prefix language."""

    candidates = _final_number_candidates(completion)
    return candidates[0] if candidates else None


def exact_match(completion: str, answer_number: int | float) -> dict[str, object]:
    """Score the extracted final number against MGSM's numeric gold answer."""

    candidates = _final_number_candidates(completion)
    prediction = answer_number if answer_number in candidates else (
        candidates[0] if candidates else None
    )
    return {
        "prediction": prediction,
        "answer_number": answer_number,
        "score": float(answer_number in candidates),
        "valid": prediction is not None,
    }


def spanish_rule_score(completion: str) -> dict[str, object]:
    """Apply AXBench's deterministic Spanish-only language rule."""

    import langdetect

    text = completion.replace("<end_of_turn>", "").strip()
    try:
        detected = langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        detected = None
    return {
        "detected_language": detected,
        "score": 2.0 if detected == "es" else 0.0,
    }
