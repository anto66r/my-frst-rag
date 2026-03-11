from embed import embed
from store import get_collection


def search(query: str, n_results: int = 3, where: dict = None) -> list[dict]:
    """
    Find the most relevant chunks for a query.

    where: optional ChromaDB filter dict, e.g.
      {"source": "WP786.pdf"}
      {"page": {"$gte": 5}}
      {"$and": [{"source": "WP786.pdf"}, {"page": {"$lte": 10}}]}
    """
    collection = get_collection()
    query_vector = embed([query])[0]

    kwargs = dict(query_embeddings=[query_vector], n_results=n_results)
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        chunks.append({
            "text": doc,
            "distance": results["distances"][0][i],
            "source": meta.get("source", "unknown"),
            "page": meta.get("page"),
        })
    return chunks


if __name__ == "__main__":
    import sys
    import json

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the effects of climate change in Ireland?"

    print(f"Query: {query}\n")

    # Example: search all docs
    print("=== All documents ===")
    for r in search(query):
        print(f"  [{r['source']} p.{r['page']}] distance={r['distance']:.4f}")
        print(f"  {r['text'][:120]}...")
        print()

    # Example: filter to a specific file
    print("=== Filtered to WP786.pdf ===")
    for r in search(query, where={"source": "WP786.pdf"}):
        print(f"  [p.{r['page']}] distance={r['distance']:.4f}")
        print(f"  {r['text'][:120]}...")
        print()
