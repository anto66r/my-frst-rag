"""
Query the persisted index. Run llama_index_build.py first to build it.
"""
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

INDEX_DIR = "./llama_db"

Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.llm = Anthropic(model="claude-opus-4-6", max_tokens=1024)

print("Loading index from disk...")
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context)
print("Ready.\n")

# Fetch 10 candidates via vector search, rerank to top 3
# Same two-stage pattern as our scratch build — just one line here
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3,
)
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)

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
