from __future__ import annotations

import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Iterable

from app.configs.codegen import (
    BOXED_PATTERN,
    LATEX_FRAC_PATTERN,
    NUMERIC_FRAGMENT_PATTERN,
    PUBLIC_QUESTION_HIDDEN_METADATA_KEYS,
    TEXT_TRANSLATION,
)


def canonical_text(value: object, *, lowercase: bool = False) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.translate(TEXT_TRANSLATION)
    text = " ".join(text.split()).strip()
    return text.casefold() if lowercase else text


def normalize_answer_text(value: object) -> str:
    return canonical_text(value, lowercase=True)


def _parse_numeric_fragment(candidate: str) -> str | None:
    compact = canonical_text(candidate).replace(",", "").replace(" ", "")
    if not compact:
        return None
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


def canonical_numeric_text(value: object) -> str | None:
    text = canonical_text(value)
    if not text:
        return None

    direct = _parse_numeric_fragment(text)
    if direct is not None:
        return direct

    boxed_matches = BOXED_PATTERN.findall(text)
    for candidate in boxed_matches:
        parsed = _parse_numeric_fragment(candidate)
        if parsed is not None:
            return parsed

    latex_fraction_matches = LATEX_FRAC_PATTERN.findall(text)
    for numerator, denominator in latex_fraction_matches:
        parsed = _parse_numeric_fragment(f"{numerator}/{denominator}")
        if parsed is not None:
            return parsed

    fragments = NUMERIC_FRAGMENT_PATTERN.findall(text)
    for fragment in reversed(fragments):
        parsed = _parse_numeric_fragment(fragment)
        if parsed is not None:
            return parsed
    return None


def answer_aliases(*values: object) -> set[str]:
    aliases = {normalize_answer_text(value) for value in values}
    aliases.discard("")
    return aliases


def choice_answer_matches(
    actual: object,
    *,
    expected: object,
    choices: Iterable[object],
    answer_alias_list: Iterable[object] = (),
    correct_choice_index: int | None = None,
) -> tuple[bool, str]:
    actual_text = normalize_answer_text(actual)
    if not actual_text:
        return False, actual_text

    aliases = answer_aliases(expected, *answer_alias_list)
    if actual_text in aliases:
        return True, actual_text

    normalized_choices = [normalize_answer_text(choice) for choice in choices]
    if correct_choice_index is None:
        return False, actual_text
    if correct_choice_index < 0 or correct_choice_index >= len(normalized_choices):
        return False, actual_text

    expected_choice = normalized_choices[correct_choice_index]
    if not expected_choice:
        return False, actual_text

    index_aliases = {
        str(correct_choice_index + 1),
        chr(ord("a") + correct_choice_index),
        f"option {correct_choice_index + 1}",
        f"choice {correct_choice_index + 1}",
        f"option {chr(ord('a') + correct_choice_index)}",
        f"choice {chr(ord('a') + correct_choice_index)}",
        f"({chr(ord('a') + correct_choice_index)})",
    }
    if actual_text in index_aliases:
        return True, expected_choice

    if actual_text == expected_choice:
        return True, expected_choice

    mentioned_choices = {
        choice
        for choice in normalized_choices
        if choice and re.search(rf"(?<!\\w){re.escape(choice)}(?!\\w)", actual_text)
    }
    if mentioned_choices == {expected_choice}:
        return True, expected_choice
    return False, actual_text


def public_question_payload(item: object) -> dict[str, object]:
    if not isinstance(item, dict):
        return {}
    metadata = {
        str(key): value
        for key, value in dict(item.get("metadata") or {}).items()
        if str(key) not in PUBLIC_QUESTION_HIDDEN_METADATA_KEYS
    }
    prompt = item.get("raw_prompt") if item.get("raw_prompt") is not None else item.get("prompt")
    context = item.get("raw_context") if item.get("raw_context") is not None else item.get("context")
    choices = item.get("raw_choices") or item.get("choices") or []
    payload: dict[str, object] = {
        "id": item.get("id") or item.get("item_id"),
        "item_id": item.get("item_id"),
        "question_id": item.get("question_id") or item.get("item_id"),
        "name": item.get("name"),
        "prompt": prompt,
        "raw_prompt": item.get("raw_prompt"),
        "context": context,
        "raw_context": item.get("raw_context"),
        "choices": list(choices),
        "raw_choices": list(item.get("raw_choices") or []),
        "metadata": metadata,
    }
    return {key: value for key, value in payload.items() if value is not None}


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
