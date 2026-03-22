from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Iterable


def canonical_numeric_text(value: object) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    compact = text.replace(",", "")
    if "/" in compact and compact.count("/") == 1:
        left, right = compact.split("/", 1)
        try:
            fraction = Fraction(int(left.strip()), int(right.strip()))
        except (ValueError, ZeroDivisionError):
            return None
        return f"{fraction.numerator}/{fraction.denominator}" if fraction.denominator != 1 else str(fraction.numerator)
    try:
        decimal_value = Decimal(compact)
    except InvalidOperation:
        return None
    normalized = decimal_value.normalize()
    if normalized == normalized.to_integral():
        return str(int(normalized))
    text_value = format(normalized, "f").rstrip("0").rstrip(".")
    return text_value or "0"


def normalize_answer_set(value: object) -> list[str]:
    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()
        if not text:
            items = []
        else:
            for separator in ("\n", ";", ",", "|"):
                text = text.replace(separator, "\n")
            items = [part.strip() for part in text.splitlines()]
    normalized: list[str] = []
    for item in items:
        candidate = str(item).strip()
        if candidate:
            normalized.append(candidate.lower())
    return normalized


def exact_set_match(actual: Iterable[str], expected: Iterable[str]) -> bool:
    return Counter(actual) == Counter(expected)


def set_f1(actual: Iterable[str], expected: Iterable[str]) -> float:
    actual_counter = Counter(actual)
    expected_counter = Counter(expected)
    if not actual_counter and not expected_counter:
        return 1.0
    overlap = sum((actual_counter & expected_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(actual_counter.values())
    recall = overlap / sum(expected_counter.values())
    return 2 * precision * recall / (precision + recall)
