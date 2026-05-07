"""PDF parsing using PyMuPDF (fitz).

extract_text() is the only public function. It accepts raw PDF bytes and
returns a list of page dicts, one per page. All PyMuPDF-specific code lives
here — swap the parser by replacing this file without touching anything else.
"""

from typing import TypedDict

import fitz  # PyMuPDF


class PageDict(TypedDict):
    page_number: int   # 1-indexed
    text: str          # extracted plain text for this page
    char_count: int    # length of text — useful for skipping blank pages


def extract_text(pdf_bytes: bytes) -> list[PageDict]:
    """Parse a PDF from raw bytes and return one dict per page.

    Blank pages (fewer than 10 characters) are included but easily filtered
    downstream — callers can check char_count > 0.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        List of PageDict, ordered by page number.

    Raises:
        ValueError: If the bytes do not represent a valid PDF.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    pages: list[PageDict] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")  # plain text, no layout preservation
        pages.append(
            PageDict(
                page_number=i,
                text=text.strip(),
                char_count=len(text.strip()),
            )
        )

    doc.close()
    return pages
