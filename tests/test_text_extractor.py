from pathlib import Path

from pipeline.text_extractor import extract_document

SAMPLES = Path(__file__).parents[1] / "samples"


def test_rent_notice_is_text_based():
    doc = extract_document(SAMPLES / "rent_notice.pdf")
    assert doc.page_count == 1
    assert "Rent" in doc.text
    assert not doc.needs_multimodal  # born-digital -> no image fallback


def test_lease_extracts_all_pages_as_text():
    doc = extract_document(SAMPLES / "lease_commercial.pdf")
    assert doc.page_count == 70
    assert "[page 1]" in doc.text and "[page 70]" in doc.text
    assert len(doc.text) > 50_000
