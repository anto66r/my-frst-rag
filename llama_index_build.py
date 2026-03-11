"""
Run this once per document set to build and save the index to disk.
Subsequent queries load from disk instantly — no re-embedding needed.
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

INDEX_DIR = "./llama_db"

Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.llm = Anthropic(model="claude-opus-4-6", max_tokens=1024)

# Smaller chunks = individual facts don't get diluted by surrounding text.
# chunk_size is in tokens (~4 chars each). 256 tokens ≈ 200 words.
# chunk_overlap carries sentences across boundaries (same idea as our overlap).
Settings.text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)

print("Loading documents...")
documents = SimpleDirectoryReader("sourcePdfs").load_data()
print(f"  {len(documents)} pages across {len({d.metadata['file_name'] for d in documents})} files")

print("Building index (chunking + embedding)...")
index = VectorStoreIndex.from_documents(documents)

print(f"Saving index to {INDEX_DIR}...")
index.storage_context.persist(persist_dir=INDEX_DIR)
print("Done.")
