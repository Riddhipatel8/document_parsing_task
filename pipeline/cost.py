"""Token / cost accounting and a baseline comparison.

We measure real usage from the LLM responses, and separately *estimate* what the
per-key-all-images baseline would have cost on the same document, so the README
can show a concrete side-by-side number rather than a hand-wave.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import litellm

from .llm_client import LLMResponse

logger = logging.getLogger(__name__)

# Gemini bills a flat ~258 tokens per image (per tile) for typical page-sized
# images. Used only for the baseline *estimate*; documented as an assumption.
IMAGE_TOKENS_PER_PAGE = 258


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    records: list[dict] = field(default_factory=list)

    def add(self, label: str, resp: LLMResponse) -> None:
        self.calls += 1
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens
        self.cached_tokens += resp.cached_tokens
        try:
            call_cost = litellm.completion_cost(completion_response=resp.raw)
        except Exception as e:
            logger.warning("Cost calculation failed for %r: %s — defaulting to $0", label, e)
            call_cost = 0.0
        self.cost_usd += call_cost or 0.0
        self.records.append(
            {
                "label": label,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "cached_tokens": resp.cached_tokens,
                "cost_usd": round(call_cost or 0.0, 6),
            }
        )
        logger.debug(
            "Usage [%s]: prompt=%d completion=%d cached=%d cost=$%.6f | "
            "running total: calls=%d prompt=%d cost=$%.6f",
            label,
            resp.prompt_tokens,
            resp.completion_tokens,
            resp.cached_tokens,
            call_cost or 0.0,
            self.calls,
            self.prompt_tokens,
            self.cost_usd,
        )

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_cost_usd": round(self.cost_usd, 6),
            "per_call": self.records,
        }


def estimate_baseline(
    num_fields: int, page_count: int, doc_text_tokens: int
) -> dict:
    """Estimate the per-key-all-images baseline on the same document.

    Baseline sends, for EACH field: all page images + that field's prompt. The
    dominant term is images re-sent once per field.
    """
    image_tokens = page_count * IMAGE_TOKENS_PER_PAGE
    baseline_prompt_tokens = num_fields * image_tokens
    logger.debug(
        "Baseline estimate: %d fields x %d pages x %d img-tokens/page = %d total tokens",
        num_fields,
        page_count,
        IMAGE_TOKENS_PER_PAGE,
        baseline_prompt_tokens,
    )
    return {
        "assumed_image_tokens_per_page": IMAGE_TOKENS_PER_PAGE,
        "calls": num_fields,
        "estimated_input_tokens": baseline_prompt_tokens,
        "note": (
            "Baseline = one call per field, re-sending all "
            f"{page_count} page images each time ({num_fields} x "
            f"{image_tokens} img-tokens)."
        ),
    }
