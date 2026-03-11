import anthropic
from search import search
from rerank import rerank


def ask(question: str, n_results: int = 3, where: dict = None) -> str:
    """Retrieve relevant chunks and ask Claude to answer based on them."""

    # Step 1: fetch more candidates than needed, then rerank to top n_results
    candidates = search(question, n_results=10, where=where)
    chunks = rerank(question, candidates, top_k=n_results)

    # Label each chunk with source + page so Claude can cite it
    context_parts = []
    for c in chunks:
        context_parts.append(f"[Source: {c['source']}, page {c['page']}]\n{c['text']}")
    context = "\n\n".join(context_parts)

    # Step 2: build the prompt — context first, then the question
    prompt = f"""Answer the question using only the context provided below.
Cite the source and page number where relevant (e.g. "According to WP786.pdf, page 3...").
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}"""

    # Step 3: call Claude
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the effects of climate change in Ireland?"

    print(f"Question: {question}\n")
    answer = ask(question)
    print(f"Answer:\n{answer}")
