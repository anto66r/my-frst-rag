"""
Evaluation harness for the LlamaIndex pipeline.
Same test set and scoring logic as evaluate.py, but uses the
hybrid retriever + reranker instead of the scratch pipeline.
"""
import anthropic
from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic as LlamaAnthropic
from llama_index.retrievers.bm25 import BM25Retriever
from eval_data import TEST_SET

INDEX_DIR = "./llama_db"

# ── Build the pipeline once ───────────────────────────────────────────────────

Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.llm = LlamaAnthropic(model="claude-opus-4-6", max_tokens=1024)

print("Loading index...")
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context)
nodes = list(index.docstore.docs.values())

vector_retriever = index.as_retriever(similarity_top_k=10)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
)
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3,
)
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker],
)

print("Ready.\n")


# ── Retrieval evaluation ──────────────────────────────────────────────────────

def retrieval_hit(question: str, keywords: list[str]) -> bool:
    nodes = hybrid_retriever.retrieve(question)
    # Apply reranker manually
    from llama_index.core.schema import QueryBundle
    reranked = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(question))
    combined = " ".join(n.text.lower() for n in reranked)
    return all(kw.lower() in combined for kw in keywords)


def evaluate_retrieval() -> dict:
    hits = 0
    results = []
    for item in TEST_SET:
        hit = retrieval_hit(item["question"], item["keywords"])
        hits += int(hit)
        results.append({"question": item["question"], "hit": hit})
    return {"recall": hits / len(TEST_SET), "hits": hits, "total": len(TEST_SET), "results": results}


# ── Answer evaluation (LLM-as-judge) ─────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating whether a RAG system's answer correctly addresses a question.

Question: {question}
Expected answer: {expected}
System answer: {actual}

Does the system answer convey the same key information as the expected answer?
Reply with exactly one word: YES or NO, then a brief reason on the same line."""


def judge_answer(question: str, expected: str, actual: str) -> dict:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, expected=expected, actual=actual
        )}]
    )
    verdict = response.content[0].text.strip()
    return {"passed": verdict.upper().startswith("YES"), "verdict": verdict}


def evaluate_answers() -> dict:
    passed = 0
    results = []
    for item in TEST_SET:
        response = query_engine.query(item["question"])
        judgment = judge_answer(item["question"], item["answer"], str(response))
        passed += int(judgment["passed"])
        results.append({
            "question": item["question"],
            "expected": item["answer"],
            "actual": str(response),
            "passed": judgment["passed"],
            "verdict": judgment["verdict"],
        })
    return {"accuracy": passed / len(TEST_SET), "passed": passed, "total": len(TEST_SET), "results": results}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("RETRIEVAL EVALUATION (recall@3)")
    print("=" * 60)
    retrieval = evaluate_retrieval()
    for r in retrieval["results"]:
        print(f"  [{'PASS' if r['hit'] else 'FAIL'}] {r['question'][:70]}")
    print(f"\nRecall@3: {retrieval['hits']}/{retrieval['total']} = {retrieval['recall']:.0%}\n")

    print("=" * 60)
    print("ANSWER EVALUATION (LLM-as-judge)")
    print("=" * 60)
    answers = evaluate_answers()
    for r in answers["results"]:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['question'][:70]}")
        if not r["passed"]:
            print(f"         Expected : {r['expected']}")
            print(f"         Got      : {r['actual'][:120]}")
        print(f"         Verdict  : {r['verdict']}")
        print()
    print(f"Accuracy: {answers['passed']}/{answers['total']} = {answers['accuracy']:.0%}")
