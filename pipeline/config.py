"""Central configuration. Everything tunable lives here so the rest of the
pipeline stays declarative and provider-agnostic."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    # Model is a plain LiteLLM string, so swapping providers is a one-line change:
    #   gemini/gemini-2.5-flash | gemini/gemini-2.5-pro | anthropic/claude-... | gpt-4o ...
    model: str = os.getenv("EXTRACTION_MODEL", "gemini/gemini-2.5-flash")

    # Consistency levers. temperature=0 + a fixed seed are the two biggest knobs
    # for making repeated runs on the same input reproducible.
    temperature: float = float(os.getenv("EXTRACTION_TEMPERATURE", "0"))
    seed: int = int(os.getenv("EXTRACTION_SEED", "42"))

    # A page yielding fewer than this many extractable characters is treated as
    # scanned/image-only and routed to the multimodal (image) fallback instead.
    min_chars_per_page: int = 20

    # One automatic repair attempt when the model's output fails schema validation.
    max_repair_attempts: int = 1

    # Request timeout (seconds) per LLM call.
    request_timeout: int = 120


SETTINGS = Settings()

logger.debug(
    "Settings loaded: model=%s temperature=%s seed=%d min_chars=%d "
    "max_repair=%d timeout=%ds",
    SETTINGS.model,
    SETTINGS.temperature,
    SETTINGS.seed,
    SETTINGS.min_chars_per_page,
    SETTINGS.max_repair_attempts,
    SETTINGS.request_timeout,
)
