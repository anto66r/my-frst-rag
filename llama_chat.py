"""
Conversational RAG with LlamaIndex's CondenseQuestionChatEngine.

Before each retrieval, the engine rewrites follow-up questions into
standalone queries so retrieval works correctly even with short,
context-dependent follow-ups like "tell me more" or "what about costs?".
"""
from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.core.chat_engine import CondenseQuestionChatEngine
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
nodes = list(index.docstore.docs.values())

# Hybrid retriever + reranker (same as llama_hybrid.py)
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

# CondenseQuestionChatEngine wraps the query engine with memory.
# On each turn it:
#   1. Rewrites the user's message + history into a standalone question
#   2. Runs that question through the query engine (retrieval + generation)
#   3. Stores the exchange in chat_history for the next turn
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    verbose=True,  # shows the rewritten question — useful for learning
)

print("RAG Chat — type 'quit' to exit, 'clear' to reset history\n")

while True:
    question = input("You: ").strip()
    if not question:
        continue
    if question.lower() == "quit":
        break
    if question.lower() == "clear":
        chat_engine.reset()
        print("History cleared.\n")
        continue

    response = chat_engine.chat(question)
    print(f"\nAssistant: {response}\n")
