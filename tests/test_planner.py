from pathlib import Path

from pipeline.document_type import get_document_type
from pipeline.planner import plan_calls

TYPES = Path(__file__).parents[1] / "samples" / "document_types.json"


def test_commercial_lease_collapses_to_four_calls():
    dt = get_document_type(TYPES, "Lease: Commercial")
    calls = plan_calls(dt)
    # 5 general fields batched into 1 + 3 individual fields = 4 calls (was 8).
    assert len(calls) == 4
    assert calls[0].label == "general"
    assert calls[0].is_batched
    assert {c.label for c in calls[1:]} == {
        "consideration", "consideration_review", "key_dates"
    }


def test_general_batch_schema_requires_all_general_fields():
    dt = get_document_type(TYPES, "Notice: Rent Change")
    batch = plan_calls(dt)[0]
    assert batch.schema["type"] == "object"
    assert set(batch.schema["required"]) == set(batch.field_names)
    assert batch.schema["additionalProperties"] is False


def test_plan_is_deterministic():
    dt = get_document_type(TYPES, "Lease: Commercial")
    assert [c.label for c in plan_calls(dt)] == [c.label for c in plan_calls(dt)]
