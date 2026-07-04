"""Parse + schema-validate model output, with one automatic repair pass.

Local validation (not just the provider's JSON mode) is the real guarantee that
output conforms to the document type's schema - including `enum`, `required` and
`additionalProperties: false`, which native structured-output modes often ignore.
It's also provider-agnostic.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

logger = logging.getLogger(__name__)


@dataclass
class ValidationOutcome:
    ok: bool
    value: Any = None
    error: str = ""


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences a model may add despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        logger.debug("Stripping markdown fences from model output")
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: -3]
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    return t.strip()


def parse_and_validate(text: str, schema: dict[str, Any]) -> ValidationOutcome:
    logger.debug("Parsing model output (%d chars)", len(text))
    try:
        value = json.loads(_strip_fences(text))
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed: %s | raw text: %.200s", e, text)
        return ValidationOutcome(ok=False, error=f"Invalid JSON: {e}")

    if not schema:
        logger.debug("No schema provided — skipping validation, returning raw value")
        return ValidationOutcome(ok=True, value=value)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(value), key=lambda e: list(e.path))
    if errors:
        msg = "; ".join(
            f"at {'/'.join(map(str, e.path)) or '<root>'}: {e.message}"
            for e in errors[:10]
        )
        logger.warning(
            "Schema validation failed (%d error(s)): %s", len(errors), msg
        )
        return ValidationOutcome(ok=False, value=value, error=msg)

    logger.debug("Schema validation passed")
    return ValidationOutcome(ok=True, value=value)


def repair_message(bad_text: str, error: str) -> list[dict[str, str]]:
    """Feed the failed output + validation error back for a single fix attempt."""
    logger.debug("Building repair message for error: %s", error)
    return [
        {"role": "assistant", "content": bad_text},
        {
            "role": "user",
            "content": (
                "Your previous response failed schema validation with these "
                f"errors:\n{error}\n\nReturn a corrected JSON value that fully "
                "conforms to the schema. Output JSON only."
            ),
        },
    ]
