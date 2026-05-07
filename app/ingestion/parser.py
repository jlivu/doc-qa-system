"""PDF parsing using PyMuPDF (fitz) with pytesseract OCR fallback.

extract_text() is the only public function. It accepts raw PDF bytes and
returns a list of page dicts, one per page.
"""

from typing import TypedDict

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from app.ingestion.exceptions import InvalidPDFError, EmptyPDFError


class PageDict(TypedDict):
    page_number: int   # 1-indexed
    text: str          # extracted plain text, stripped
    char_count: int    # len(text) — used to detect blank pages
    ocr_used: bool     # True if pytesseract was used for this page


def extract_text(pdf_bytes: bytes) -> list[PageDict]:
    """Parse a PDF from raw bytes and return one dict per page.

    For each page, native text extraction is attempted first. If the
    result has fewer than 20 characters, the page is rasterised at
    300 DPI grayscale and passed through pytesseract OCR.

    Blank pages are included (with char_count == 0), not omitted.

    Raises:
        InvalidPDFError: If the bytes cannot be opened as a PDF.
        EmptyPDFError: If the document has zero pages.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise InvalidPDFError(f"Could not open PDF: {exc}") from exc

    try:
        if doc.page_count == 0:
            raise EmptyPDFError("PDF has zero pages")

        pages: list[PageDict] = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            ocr_used = False

            if len(text) < 20:
                pix = page.get_pixmap(dpi=300, colorspace=fitz.csGRAY)
                img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img, lang="eng").strip()
                if ocr_text:
                    text = ocr_text
                    ocr_used = True

            pages.append(
                PageDict(
                    page_number=i,
                    text=text,
                    char_count=len(text),
                    ocr_used=ocr_used,
                )
            )

        return pages
    finally:
        doc.close()
