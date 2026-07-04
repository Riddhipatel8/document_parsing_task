"""Provider-agnostic LLM wrapper built on LiteLLM.

Why LiteLLM: the model becomes a config string, so Gemini/Claude/GPT are all the
same code path. It also normalizes the two things the pipeline depends on -
JSON-mode output and token `usage` accounting - across providers.

Consistency: every call uses temperature=0 + a fixed seed and a byte-identical,
leading document block. Keeping that block identical across the ~4 calls maximizes
Gemini 2.5's *implicit* prompt caching (repeated prefixes are auto-discounted), so
reusing the document is cheap without any provider-specific cache code.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import litellm

from .config import SETTINGS
from .text_extractor import DocumentContent

logger = logging.getLogger(__name__)

# Unsupported params (e.g. seed on some providers) are dropped instead of raising,
# keeping the client truly provider-agnostic.
litellm.drop_params = True

SYSTEM_INSTRUCTIONS = (
    "You are a precise document-extraction engine for property/lease documents. "
    "Extract values ONLY from the provided document. Never invent facts. When a "
    "value is absent, use an empty string or empty array as the schema allows. "
    "Return a single valid JSON value that conforms to the requested JSON Schema, "
    "with no markdown fences or commentary. Be deterministic: given the same input, "
    "always produce the same output."
)


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    model: str
    raw: Any = None


class LLMClient:
    def __init__(self, model: str | None = None):
        self.model = model or SETTINGS.model
        logger.debug("LLMClient initialised: model=%s", self.model)

    def _document_message(self, doc: DocumentContent) -> dict[str, Any]:
        """The large, shared, cache-friendly document block (identical per call)."""
        header = "=== DOCUMENT START ===\n"
        footer = "\n=== DOCUMENT END ==="
        if not doc.needs_multimodal:
            logger.debug("Document message: text-only (%d chars)", len(doc.text))
            return {"role": "user", "content": header + doc.text + footer}

        logger.debug(
            "Document message: multimodal (%d chars text + %d image page(s))",
            len(doc.text),
            len(doc.image_pages),
        )
        parts: list[dict[str, Any]] = [{"type": "text", "text": header + doc.text}]
        for b64 in doc.image_pages:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        parts.append({"type": "text", "text": footer})
        return {"role": "user", "content": parts}

    @staticmethod
    def _instruction_message(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        content = prompt.strip()
        if schema:
            content += (
                "\n\nReturn ONLY a JSON value conforming to this JSON Schema:\n"
                + json.dumps(schema)
            )
        return {"role": "user", "content": content}

    def complete(
        self,
        doc: DocumentContent,
        prompt: str,
        schema: dict[str, Any],
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            self._document_message(doc),  # stable leading block -> implicit cache
            self._instruction_message(prompt, schema),
        ]
        if extra_messages:
            logger.debug("Appending %d extra message(s) (repair path)", len(extra_messages))
            messages.extend(extra_messages)

        logger.info(
            "LLM call: model=%s messages=%d schema_keys=%s",
            self.model,
            len(messages),
            list(schema.get("properties", schema).keys()) if schema else "none",
        )

        resp = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=SETTINGS.temperature,
            seed=SETTINGS.seed,
            response_format={"type": "json_object"},
            timeout=SETTINGS.request_timeout,
        )

        usage = resp.usage
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0

        llm_resp = LLMResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_tokens=cached,
            model=self.model,
            raw=resp,
        )

        logger.info(
            "LLM response: prompt_tokens=%d completion_tokens=%d cached_tokens=%d "
            "response_len=%d chars",
            llm_resp.prompt_tokens,
            llm_resp.completion_tokens,
            llm_resp.cached_tokens,
            len(llm_resp.text),
        )
        logger.debug("LLM raw response text: %.500s%s",
                     llm_resp.text,
                     "..." if len(llm_resp.text) > 500 else "")

        return llm_resp
