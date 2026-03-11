import anthropic
from search import search
from rerank import rerank

# How many chunks to retrieve and rerank per turn
N_CANDIDATES = 10
TOP_K = 3

# Maximum number of past conversation turns to keep in the prompt.
# Each turn = one user message + one assistant message.
# Older turns are dropped to avoid hitting the context window limit.
MAX_HISTORY_TURNS = 6


def build_context(question: str, where: dict = None) -> str:
    """Retrieve and rerank chunks relevant to the current question."""
    candidates = search(question, n_results=N_CANDIDATES, where=where)
    chunks = rerank(question, candidates, top_k=TOP_K)
    parts = [f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" for c in chunks]
    return "\n\n".join(parts)


def chat(where: dict = None):
    """
    Interactive REPL. Each turn:
    1. Retrieve fresh context relevant to the new question
    2. Inject it into the system prompt
    3. Send the full conversation history to Claude
    4. Print the response and append it to history

    The system prompt is updated every turn with new context so Claude
    always has relevant chunks for the current question, while the
    message history gives it memory of the conversation so far.
    """
    client = anthropic.Anthropic()
    history = []  # list of {"role": "user"|"assistant", "content": "..."}

    print("RAG Chat — type 'quit' to exit, 'clear' to reset history\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "clear":
            history = []
            print("History cleared.\n")
            continue

        # Fresh retrieval for every question
        context = build_context(question, where=where)

        system_prompt = f"""You are a helpful assistant that answers questions about documents.
Answer using only the context provided below. Cite the source and page number where relevant.
If the answer is not in the context, say you don't have enough information.

Context:
{context}"""

        # Trim history to avoid exceeding the context window
        trimmed_history = history[-(MAX_HISTORY_TURNS * 2):]

        history.append({"role": "user", "content": question})

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=trimmed_history + [{"role": "user", "content": question}],
        )

        answer = response.content[0].text
        history.append({"role": "assistant", "content": answer})

        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    chat()
