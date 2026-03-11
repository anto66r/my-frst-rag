import os
import chromadb
from embed import embed

# PersistentClient saves to disk so you don't re-embed every time
client = chromadb.PersistentClient(path="./db")


def get_collection(name: str = "docs"):
    """Get or create a collection (like a table in a regular DB)."""
    return client.get_or_create_collection(name)


def ingest(pdf_path: str, collection_name: str = "docs"):
    """Extract, chunk, embed, and store a PDF."""
    from extract import extract_pages
    from chunk import chunk_pages

    print(f"Reading {pdf_path}...")
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages)
    print(f"  {len(pages)} pages → {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]

    print("Embedding...")
    vectors = embed(texts)

    print("Storing in ChromaDB...")
    collection = get_collection(collection_name)
    filename = os.path.basename(pdf_path)

    collection.add(
        ids=[f"{pdf_path}-chunk-{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {"source": filename, "page": c["page"], "chunk": i}
            for i, c in enumerate(chunks)
        ],
    )
    print(f"Done. Collection now has {collection.count()} chunks.")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "WP786.pdf"
    ingest(path)
