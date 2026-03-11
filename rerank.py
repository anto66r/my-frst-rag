from sentence_transformers import CrossEncoder

# Loads ~80MB on first run, cached after that
_model = None


def get_reranker():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Re-score a list of chunks against the query using a cross-encoder,
    then return the top_k by score.

    The cross-encoder sees the query and each chunk together — much more
    accurate than the bi-encoder used in vector search, but too slow to
    run against the entire DB. Use it as a second-pass filter.

    Each input chunk dict must have a "text" key.
    Each output chunk dict gains a "rerank_score" key (higher = more relevant).
    """
    model = get_reranker()
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    from search import search

    query = "What is the predicted temperature rise in Ireland?"
    print(f"Query: {query}\n")

    # Fetch more candidates than we need
    candidates = search(query, n_results=10)

    print(f"Top 10 from vector search:")
    for i, c in enumerate(candidates):
        print(f"  {i+1}. [p.{c['page']}] distance={c['distance']:.4f} | {c['text'][:80]}...")

    print(f"\nTop 3 after reranking:")
    reranked = rerank(query, candidates, top_k=3)
    for i, c in enumerate(reranked):
        print(f"  {i+1}. [p.{c['page']}] rerank_score={c['rerank_score']:.4f} | {c['text'][:80]}...")
