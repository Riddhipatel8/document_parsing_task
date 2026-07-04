"""Turn a document type's fields into a minimal set of LLM calls.

This is the core cost decision. The baseline makes **one call per key** and
re-sends the whole document every time. We instead group:

* all ``general`` fields  -> ONE batched call (they're simple strings sharing
  context; no reason to spend a call each);
* each ``individual`` field -> its OWN call (each carries a huge dedicated prompt
  and deep schema; merging them would blow context and cross-contaminate).

So an 8-field commercial lease goes from 8 calls to ~4, and every call reuses the
same (cacheable) document context. Ordering is deterministic for reproducibility.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .document_type import DocumentType, Field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallSpec:
    """One planned LLM call: which fields, what to ask, what shape to return."""

    label: str  # "general" or the individual field name
    field_names: list[str]
    prompt: str
    schema: dict[str, Any]
    is_batched: bool


def _general_batch_schema(fields: list[Field]) -> dict[str, Any]:
    """A strict object schema requiring every general field as its item_type."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [f.name for f in fields],
        "properties": {
            f.name: {
                "type": f.item_type or "string",
                "description": f.description or f.name,
            }
            for f in fields
        },
    }


def _general_batch_prompt(fields: list[Field]) -> str:
    lines = [
        "Extract the following fields from the document. Return one JSON object "
        "with exactly these keys. If a value is not present in the document, use "
        "an empty string. Do not guess.",
        "",
    ]
    for f in fields:
        lines.append(f"- {f.name}: {f.description or f.name}")
    return "\n".join(lines)


def plan_calls(doc_type: DocumentType) -> list[CallSpec]:
    logger.info(
        "Planning calls for %r: %d general, %d individual fields",
        doc_type.name,
        len(doc_type.general_fields),
        len(doc_type.individual_fields),
    )
    calls: list[CallSpec] = []

    # 1) Batched call for all the cheap general/scalar fields.
    general = sorted(doc_type.general_fields, key=lambda f: f.name)
    if general:
        calls.append(
            CallSpec(
                label="general",
                field_names=[f.name for f in general],
                prompt=_general_batch_prompt(general),
                schema=_general_batch_schema(general),
                is_batched=True,
            )
        )
        logger.debug(
            "  [call 1] general batch: %d fields: %s",
            len(general),
            [f.name for f in general],
        )

    # 2) One dedicated call per individual field, in a stable order.
    for idx, f in enumerate(sorted(doc_type.individual_fields, key=lambda f: f.name), start=2):
        calls.append(
            CallSpec(
                label=f.name,
                field_names=[f.name],
                prompt=f.description,
                schema=f.schema,
                is_batched=False,
            )
        )
        logger.debug(
            "  [call %d] individual: %r (has_schema=%s)",
            idx, f.name, bool(f.schema),
        )

    logger.info("Call plan ready: %d call(s) total (baseline would be %d)", len(calls), len(doc_type.fields))
    return calls
