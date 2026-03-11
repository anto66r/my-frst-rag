import re


def chunk_text(text: str, max_words: int = 250) -> list[str]:
    """
    Split text into chunks by paragraph, merging short paragraphs and
    splitting long ones. This preserves natural structure rather than
    cutting arbitrarily at a word boundary.

    max_words: soft limit — a single paragraph over this size gets split
               at sentence boundaries.
    """
    # Normalize whitespace: collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    # Split into paragraphs on blank lines
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in raw_paragraphs:
        para_words = len(para.split())

        # If a single paragraph exceeds max_words, split it at sentences
        if para_words > max_words:
            # Flush whatever we've accumulated first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            sentences = re.split(r"(?<=[.!?])\s+", para)
            sentence_buffer = []
            buffer_words = 0

            for sentence in sentences:
                s_words = len(sentence.split())
                if buffer_words + s_words > max_words and sentence_buffer:
                    chunks.append(" ".join(sentence_buffer))
                    sentence_buffer = []
                    buffer_words = 0
                sentence_buffer.append(sentence)
                buffer_words += s_words

            if sentence_buffer:
                chunks.append(" ".join(sentence_buffer))

        # If adding this paragraph keeps us under the limit, accumulate it
        elif current_word_count + para_words <= max_words:
            current_chunk.append(para)
            current_word_count += para_words

        # Otherwise flush the current chunk and start a new one
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_word_count = para_words

    # Flush any remaining content
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_pages(pages: list[dict], max_words: int = 250) -> list[dict]:
    """
    Chunk a list of pages (from extract_pages), carrying page number forward
    into each chunk's metadata.

    Each returned dict has:
      - text: the chunk content
      - page: the page number the chunk started on
    """
    chunks = []
    for page in pages:
        page_chunks = chunk_text(page["text"], max_words=max_words)
        for chunk in page_chunks:
            chunks.append({"text": chunk, "page": page["page"]})
    return chunks


if __name__ == "__main__":
    from extract import extract_pages
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "WP786.pdf"
    pages = extract_pages(path)
    chunks = chunk_pages(pages)

    print(f"Pages  : {len(pages)}")
    print(f"Chunks : {len(chunks)}")
    print()
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} | page {chunk['page']} ({len(chunk['text'].split())} words) ---")
        print(chunk["text"])
        print()
