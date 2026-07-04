"""End-to-end smoke test with a mocked LLM (no API key / network needed)."""
from pathlib import Path
from typing import Any

from pipeline.document_type import get_document_type
from pipeline.llm_client import LLMResponse
from pipeline.pipeline import extract

SAMPLES = Path(__file__).parents[1] / "samples"
TYPES = SAMPLES / "document_types.json"


def _minimal_instance(schema: dict[str, Any]) -> Any:
    """Build the smallest value that satisfies a (subset of) JSON Schema."""
    t = schema.get("type")
    if t == "object":
        return {
            k: _minimal_instance(v)
            for k, v in schema.get("properties", {}).items()
            if k in schema.get("required", [])
        }
    if t == "array":
        return []
    if t == "number" or t == "integer":
        return 0
    if t == "boolean":
        return False
    if "enum" in schema:
        return schema["enum"][0]
    return ""


class FakeClient:
    """Returns a schema-valid, deterministic response for any call."""

    model = "fake/mock"

    def __init__(self):
        self.calls = 0

    def complete(self, doc, prompt, schema, extra_messages=None) -> LLMResponse:
        import json

        self.calls += 1
        return LLMResponse(
            text=json.dumps(_minimal_instance(schema)),
            prompt_tokens=1000,
            completion_tokens=50,
            cached_tokens=0,
            model=self.model,
        )


def test_end_to_end_rent_notice_conforms():
    dt = get_document_type(TYPES, "Notice: Rent Change")
    out = extract(SAMPLES / "rent_notice.pdf", dt, client=FakeClient())

    # No validation errors -> every field's output conformed to its schema.
    assert out.errors == []
    # All declared fields present in the merged result.
    for f in dt.fields:
        assert f.name in out.data
    # 5 general fields batched to 1 call + 2 individual = 3 calls total.
    assert out.usage["calls"] == 3
    # Baseline (7 calls, all pages re-sent per field) is more input tokens.
    assert out.baseline["calls"] == 7
