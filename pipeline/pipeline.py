"""Orchestrator: PDF + document type -> validated, merged JSON."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm

from .config import SETTINGS
from .cost import Usage, estimate_baseline
from .document_type import DocumentType
from .llm_client import LLMClient
from .planner import CallSpec, plan_calls
from .text_extractor import DocumentContent, extract_document
from .validator import parse_and_validate, repair_message

logger = logging.getLogger(__name__)


@dataclass
class ExtractionOutput:
    data: dict[str, Any]
    usage: dict
    baseline: dict
    errors: list[str] = field(default_factory=list)


def _run_call(client: LLMClient, doc: DocumentContent, call: CallSpec, usage: Usage,
              errors: list[str]) -> Any:
    logger.info("Running call: label=%r fields=%s", call.label, call.field_names)

    resp = client.complete(doc, call.prompt, call.schema)
    usage.add(call.label, resp)
    outcome = parse_and_validate(resp.text, call.schema)

    attempts = 0
    while not outcome.ok and attempts < SETTINGS.max_repair_attempts:
        attempts += 1
        logger.warning(
            "Validation failed for %r (attempt %d/%d): %s — sending repair request",
            call.label, attempts, SETTINGS.max_repair_attempts, outcome.error,
        )
        resp = client.complete(
            doc, call.prompt, call.schema,
            extra_messages=repair_message(resp.text, outcome.error),
        )
        usage.add(f"{call.label}:repair{attempts}", resp)
        outcome = parse_and_validate(resp.text, call.schema)

    if not outcome.ok:
        logger.error(
            "Call %r failed validation after %d repair attempt(s): %s",
            call.label, attempts, outcome.error,
        )
        errors.append(f"{call.label}: {outcome.error}")
    else:
        logger.debug("Call %r validated OK (repairs=%d)", call.label, attempts)

    return outcome.value


def _assign(data: dict[str, Any], call: CallSpec, value: Any) -> None:
    if value is None:
        logger.debug("Call %r returned None — skipping assignment", call.label)
        return
    if call.is_batched and isinstance(value, dict):
        # General batch returns {field_name: value}; splat into the result.
        for name in call.field_names:
            data[name] = value.get(name, "")
            logger.debug("  Assigned [general] %r = %r", name, data[name])
    else:
        data[call.field_names[0]] = value
        logger.debug("  Assigned [individual] %r = %r", call.field_names[0], value)


def extract(
    pdf_path: str | Path,
    doc_type: DocumentType,
    client: LLMClient | None = None,
) -> ExtractionOutput:
    pdf_path = Path(pdf_path)
    logger.info(
        "=== Extraction start: pdf=%s doc_type=%r ===", pdf_path, doc_type.name
    )

    client = client or LLMClient()
    doc = extract_document(pdf_path)
    logger.info(
        "Document extracted: %d pages, %d image pages, text=%d chars",
        doc.page_count, len(doc.image_pages), len(doc.text),
    )

    calls = plan_calls(doc_type)
    logger.info("Planned %d LLM call(s)", len(calls))

    usage = Usage()
    errors: list[str] = []
    data: dict[str, Any] = {}

    for call in calls:
        value = _run_call(client, doc, call, usage, errors)
        _assign(data, call, value)

    logger.info(
        "All calls complete: %d call(s), %d prompt tokens, %d cached, $%.6f",
        usage.calls, usage.prompt_tokens, usage.cached_tokens, usage.cost_usd,
    )

    try:
        doc_tokens = litellm.token_counter(model=client.model, text=doc.text)
        logger.debug("Document token count (for baseline): %d", doc_tokens)
    except Exception as e:
        doc_tokens = len(doc.text) // 4  # rough fallback
        logger.warning("token_counter failed (%s) — using rough estimate: %d", e, doc_tokens)

    baseline = estimate_baseline(
        num_fields=len(doc_type.fields),
        page_count=doc.page_count,
        doc_text_tokens=doc_tokens,
    )
    baseline["actual_calls"] = usage.calls
    baseline["actual_input_tokens"] = usage.prompt_tokens
    if baseline["estimated_input_tokens"]:
        reduction = round(
            baseline["estimated_input_tokens"] / max(usage.prompt_tokens, 1), 2
        )
        baseline["input_token_reduction_x"] = reduction
        logger.info(
            "Token reduction vs baseline: %.2fx (%d estimated → %d actual)",
            reduction,
            baseline["estimated_input_tokens"],
            usage.prompt_tokens,
        )

    if errors:
        logger.warning("Extraction finished with %d error(s): %s", len(errors), errors)
    else:
        logger.info("=== Extraction complete: %d fields extracted, no errors ===", len(data))

    return ExtractionOutput(
        data=data, usage=usage.summary(), baseline=baseline, errors=errors
    )
