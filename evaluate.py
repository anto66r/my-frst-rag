import anthropic
from search import search
from rerank import rerank
from generate import ask
from eval_data import TEST_SET


# ── Retrieval evaluation ──────────────────────────────────────────────────────

def retrieval_hit(question: str, keywords: list[str], k: int = 3) -> bool:
    """
    Check whether the top-k retrieved chunks contain all the expected keywords.
    This is a proxy for "did we retrieve the right chunk?"

    A stricter approach would be to manually label which chunk index is correct,
    but keyword matching is good enough for a first pass.
    """
    candidates = search(question, n_results=10)
    chunks = rerank(question, candidates, top_k=k)
    combined = " ".join(c["text"].lower() for c in chunks)
    return all(kw.lower() in combined for kw in keywords)


def evaluate_retrieval(k: int = 3) -> dict:
    """Run retrieval eval on the full test set and return recall@k."""
    hits = 0
    results = []

    for item in TEST_SET:
        hit = retrieval_hit(item["question"], item["keywords"], k=k)
        hits += int(hit)
        results.append({
            "question": item["question"],
            "hit": hit,
        })

    recall = hits / len(TEST_SET)
    return {"recall_at_k": recall, "k": k, "hits": hits, "total": len(TEST_SET), "results": results}


# ── Answer evaluation (LLM-as-judge) ─────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating whether a RAG system's answer correctly addresses a question.

Question: {question}
Expected answer: {expected}
System answer: {actual}

Does the system answer convey the same key information as the expected answer?
Reply with exactly one word: YES or NO, then a brief reason on the same line.
Example: YES - correctly states the 0.9°C figure and the time period."""


def judge_answer(question: str, expected: str, actual: str) -> dict:
    """Use Claude to judge whether the generated answer matches the expected answer."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question,
                expected=expected,
                actual=actual,
            )
        }]
    )
    verdict_line = response.content[0].text.strip()
    passed = verdict_line.upper().startswith("YES")
    return {"passed": passed, "verdict": verdict_line}


def evaluate_answers() -> dict:
    """Generate answers for all test questions and judge them."""
    passed = 0
    results = []

    for item in TEST_SET:
        actual = ask(item["question"])
        judgment = judge_answer(item["question"], item["answer"], actual)
        passed += int(judgment["passed"])
        results.append({
            "question": item["question"],
            "expected": item["answer"],
            "actual": actual,
            "passed": judgment["passed"],
            "verdict": judgment["verdict"],
        })

    accuracy = passed / len(TEST_SET)
    return {"accuracy": accuracy, "passed": passed, "total": len(TEST_SET), "results": results}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("RETRIEVAL EVALUATION (recall@3)")
    print("=" * 60)
    retrieval = evaluate_retrieval(k=3)
    for r in retrieval["results"]:
        status = "PASS" if r["hit"] else "FAIL"
        print(f"  [{status}] {r['question'][:70]}")
    print(f"\nRecall@3: {retrieval['hits']}/{retrieval['total']} = {retrieval['recall_at_k']:.0%}\n")

    print("=" * 60)
    print("ANSWER EVALUATION (LLM-as-judge)")
    print("=" * 60)
    answers = evaluate_answers()
    for r in answers["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['question'][:70]}")
        if not r["passed"]:
            print(f"         Expected : {r['expected']}")
            print(f"         Got      : {r['actual'][:120]}")
        print(f"         Verdict  : {r['verdict']}")
        print()
    print(f"Accuracy: {answers['passed']}/{answers['total']} = {answers['accuracy']:.0%}")
