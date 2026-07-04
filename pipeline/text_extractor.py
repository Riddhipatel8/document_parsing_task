"""PDF -> text, with a per-page multimodal fallback.

Design choice: **text-first, not image-first.** Both sample PDFs are born-digital
(the 70-page lease has ~136k chars of extractable text). Text tokens are ~10-50x
cheaper than image tokens, so we send text whenever we can and only fall back to
sending a page *image* for pages that have (almost) no extractable text - i.e.
scanned or signature pages. Gemini is natively multimodal, so those images ride
along on the same message interface with no separate OCR dependency.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from .config import SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class DocumentContent:
    """The extracted document, ready to hand to the LLM client."""

    text: str
    # base64-encoded PNGs for pages that had no usable text (scanned pages).
    image_pages: list[str] = field(default_factory=list)
    page_count: int = 0
    ocr_page_numbers: list[int] = field(default_factory=list)

    @property
    def needs_multimodal(self) -> bool:
        return bool(self.image_pages)


def extract_document(pdf_path: str | Path) -> DocumentContent:
    pdf_path = Path(pdf_path)
    logger.info("Opening PDF: %s", pdf_path)

    doc = fitz.open(pdf_path)
    logger.debug(
        "PDF opened: %d pages, min_chars_per_page=%d",
        doc.page_count,
        SETTINGS.min_chars_per_page,
    )

    text_parts: list[str] = []
    image_pages: list[str] = []
    ocr_pages: list[int] = []

    for i, page in enumerate(doc, start=1):
        page_text = page.get_text("text").strip()
        char_count = len(page_text)
        if char_count >= SETTINGS.min_chars_per_page:
            logger.debug("Page %d: %d chars — text path", i, char_count)
            # Page markers let the model cite/locate evidence ("page 12").
            text_parts.append(f"[page {i}]\n{page_text}")
        else:
            logger.debug(
                "Page %d: %d chars (below threshold=%d) — image fallback",
                i,
                char_count,
                SETTINGS.min_chars_per_page,
            )
            # Scanned/near-empty page -> render an image for the multimodal path.
            pix = page.get_pixmap(dpi=150)
            image_pages.append(base64.b64encode(pix.tobytes("png")).decode())
            ocr_pages.append(i)
            text_parts.append(f"[page {i}]\n(no extractable text - see image)")

    logger.info(
        "Extraction complete: %d/%d pages as text, %d as images (ocr pages: %s)",
        doc.page_count - len(ocr_pages),
        doc.page_count,
        len(ocr_pages),
        ocr_pages if ocr_pages else "none",
    )

    return DocumentContent(
        text="\n\n".join(text_parts),
        image_pages=image_pages,
        page_count=doc.page_count,
        ocr_page_numbers=ocr_pages,
    )
