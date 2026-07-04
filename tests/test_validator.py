from pipeline.validator import parse_and_validate

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["amount", "currency"],
    "properties": {
        "amount": {"type": "number"},
        "currency": {"type": "string", "enum": ["GBP", "USD"]},
    },
}


def test_valid_json_passes():
    out = parse_and_validate('{"amount": 100, "currency": "GBP"}', SCHEMA)
    assert out.ok and out.value["amount"] == 100


def test_strips_markdown_fences():
    out = parse_and_validate('```json\n{"amount": 1, "currency": "USD"}\n```', SCHEMA)
    assert out.ok


def test_enum_violation_fails_with_message():
    out = parse_and_validate('{"amount": 1, "currency": "EUR"}', SCHEMA)
    assert not out.ok and "currency" in out.error


def test_additional_property_rejected():
    out = parse_and_validate('{"amount": 1, "currency": "GBP", "x": 1}', SCHEMA)
    assert not out.ok


def test_malformed_json_reports_error():
    out = parse_and_validate("{not json", SCHEMA)
    assert not out.ok and "Invalid JSON" in out.error
