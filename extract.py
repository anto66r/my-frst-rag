import fitz  # pymupdf


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Return a list of pages, each with its text and page number.
    Page numbers are 1-indexed to match what you'd see in a PDF viewer.
    """
    doc = fitz.open(pdf_path)
    return [
        {"text": page.get_text(), "page": page.number + 1}
        for page in doc
    ]


def extract_text(pdf_path: str) -> str:
    """Return all text as a single string (kept for compatibility)."""
    return "\n\n".join(p["text"] for p in extract_pages(pdf_path))


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    pages = extract_pages(path)
    print(f"Extracted {len(pages)} pages")
    for p in pages[:2]:
        print(f"\n--- Page {p['page']} ---")
        print(p["text"][:300])
