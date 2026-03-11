from sentence_transformers import SentenceTransformer

# Load the model — downloads ~90MB on first run, cached after that
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts: list[str]) -> list[list[float]]:
    """Convert a list of text chunks into a list of vectors."""
    return model.encode(texts, show_progress_bar=True).tolist()


if __name__ == "__main__":
    from extract import extract_text
    from chunk import chunk_text

    text = extract_text("WP786.pdf")
    chunks = chunk_text(text)

    print(f"Embedding {len(chunks)} chunks...")
    vectors = embed(chunks)

    print(f"\nDone.")
    print(f"Each chunk → vector of {len(vectors[0])} numbers")
    print(f"\nFirst vector (first 10 numbers):")
    print([round(v, 4) for v in vectors[0][:10]])
