"""
Hybrid search: combines vector search (semantic) with BM25 (keyword).
Fixes the 60%/coast failure where the fact was buried in a chunk and
the vector search missed it entirely.
"""
from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.retrievers.bm25 import BM25Retriever

INDEX_DIR = "./llama_db"

Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.llm = Anthropic(model="claude-opus-4-6", max_tokens=1024)

print("Loading index...")
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context)

# Retrieve all stored nodes for BM25 (it needs the raw text, not vectors)
nodes = list(index.docstore.docs.values())
print(f"  {len(nodes)} nodes loaded\n")

# ── Two retrievers ────────────────────────────────────────────────────────────

# 1. Vector retriever — semantic similarity (what we've been using)
vector_retriever = index.as_retriever(similarity_top_k=10)

# 2. BM25 retriever — keyword matching (TF-IDF style)
#    This is what finds "60%" and "coast" even when the embedding misses it
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

# ── Fuse them ────────────────────────────────────────────────────────────────
# QueryFusionRetriever merges results from both retrievers,
# deduplicates, and re-scores using Reciprocal Rank Fusion (RRF).
# num_queries=1 means no query expansion — just fuse the two result sets.
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
)

# ── Reranker on top ───────────────────────────────────────────────────────────
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3,
)

query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker],
)

# ── Test ──────────────────────────────────────────────────────────────────────
questions = [
    "What is the predicted temperature rise in Ireland by mid-century?",
    "What are Ireland's targets for reducing greenhouse gas emissions by 2030?",
    "What percentage of Ireland's population lives within 10km of the coast?",
]

for question in questions:
    print(f"Q: {question}")
    response = query_engine.query(question)
    print(f"A: {response}")
    pages = [n.metadata.get("page_label", "?") for n in response.source_nodes]
    print(f"   Pages: {pages}\n")
