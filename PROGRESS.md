# RAG System — Build Progress

A from-scratch RAG (Retrieval-Augmented Generation) system in Python, built step by step.

---

## What is RAG?

RAG lets you ask questions about your own documents using an LLM. Instead of fine-tuning a model, you:
1. Break documents into chunks and store them as vectors (embeddings)
2. When a question comes in, find the most relevant chunks via vector similarity search
3. Pass those chunks as context to an LLM to generate a grounded answer

---

## Stack

| Component | Tool | Notes |
|-----------|------|-------|
| PDF parsing | `pymupdf` (fitz) | Faster and more reliable than PyPDF2 |
| Chunking | Custom Python | Paragraph-aware, sentence-boundary splitting |
| Embeddings | `sentence-transformers` | Model: `all-MiniLM-L6-v2`, 384-dim vectors, runs locally |
| Vector store | `chromadb` | Persistent local DB saved to `./db/` |
| LLM | Claude API (`claude-opus-4-6`) | Via `anthropic` Python SDK |

---

## Project Structure

```
rag/
├── venv/                  # Python virtual environment
├── db/                    # ChromaDB vector store (auto-created on ingest)
├── extract.py             # PDF → list of pages with text + page numbers
├── chunk.py               # Text → paragraph-aware chunks with page metadata
├── embed.py               # Text chunks → vectors using sentence-transformers
├── store.py               # Ingest pipeline: extract → chunk → embed → store
├── search.py              # Query → vector search → relevant chunks
├── rerank.py              # Cross-encoder reranker, second-pass filter over candidates
├── generate.py            # Question → search → rerank → Claude → answer with citations
├── chat.py                # Interactive REPL with conversation memory
└── PROGRESS.md            # This file
```

---

## Completed Steps

### Step 1 — PDF Text Extraction (`extract.py`)

`extract_pages(pdf_path)` opens a PDF with pymupdf and returns a list of dicts:
```python
[{"text": "...", "page": 1}, {"text": "...", "page": 2}, ...]
```
Page numbers are 1-indexed to match what you'd see in a PDF viewer.
`extract_text()` is also kept for simple single-string use.

### Step 2 — Chunking (`chunk.py`)

Two functions:
- `chunk_text(text, max_words=250)` — splits a single string into chunks
- `chunk_pages(pages, max_words=250)` — wraps `chunk_text`, operating page by page and carrying the page number into each chunk's metadata

**Strategy:**
- Split on blank lines (paragraph boundaries) first
- Merge short adjacent paragraphs up to `max_words`
- If a single paragraph exceeds `max_words`, split at sentence boundaries (`.`, `!`, `?`)
- This preserves natural document structure rather than cutting mid-sentence

**Why this matters:** Embedding models work best on semantically coherent units. A paragraph is usually one idea; a mid-sentence cut loses context.

### Step 3 — Embeddings (`embed.py`)

`embed(texts)` converts a list of strings into a list of 384-dimensional float vectors using `sentence-transformers` (`all-MiniLM-L6-v2`). The model runs locally — no API key needed. Downloads ~90MB on first run, cached after that.

Every chunk becomes the same-size vector regardless of its length. Chunks about similar topics end up with similar vectors — this is what makes semantic search possible.

### Step 4 — Vector Store (`store.py`)

`ingest(pdf_path)` is the full pipeline:
1. Extract pages with page numbers
2. Chunk pages into paragraph-aware chunks
3. Embed all chunk texts
4. Store in ChromaDB with metadata

Each chunk is stored with three pieces of metadata:
- `source`: the PDF filename (e.g. `WP786.pdf`)
- `page`: the page number the chunk came from
- `chunk`: the chunk index within the document

ChromaDB saves to `./db/` on disk — you only need to run ingest once per document.

To ingest a document:
```bash
python store.py yourfile.pdf
```

To re-ingest from scratch (e.g. after changing chunking logic):
```bash
rm -rf db/
python store.py yourfile.pdf
```

### Step 5 — Search (`search.py`)

`search(query, n_results=3, where=None)` finds the most relevant chunks:
1. Embeds the query using the same model as the chunks
2. Asks ChromaDB for the `n_results` closest vectors
3. Returns a list of dicts with `text`, `distance`, `source`, and `page`

**Lower distance = more similar to the query.**

The optional `where` parameter filters results before the vector search using ChromaDB's filter syntax:
```python
search(query, where={"source": "WP786.pdf"})          # single document
search(query, where={"page": {"$gte": 5}})             # page 5 onwards
search(query, where={"page": {"$lte": 3}})             # first 3 pages only
search(query, where={"$and": [{"source": "WP786.pdf"}, {"page": 5}]})
```

### Step 6 — Generation (`generate.py`)

`ask(question, n_results=3, where=None)` is the full RAG pipeline:
1. Calls `search()` to retrieve the top chunks
2. Formats them with `[Source: filename, page N]` labels
3. Sends context + question to Claude with instructions to cite sources
4. Returns Claude's answer

The prompt instructs Claude to:
- Answer using only the provided context
- Cite source filename and page number
- Say "I don't have enough information" if the answer isn't in the context

Usage:
```bash
python generate.py "What are Ireland's emissions targets?"
```

Or programmatically with filtering:
```python
from generate import ask
answer = ask("What are the flood risks?", where={"source": "WP786.pdf"})
answer = ask("What does page 5 discuss?", where={"page": 5})
```

---

## Completed Steps (continued)

### Step 7 — Reranking (`rerank.py`)

**Problem:** Vector search finds chunks that are *similar* to the query, but similarity doesn't always equal relevance. A chunk might use the same words without actually answering the question.

**Solution:** Two-stage retrieval — vector search fetches 10 candidates fast, a cross-encoder reranker picks the best 3.

`rerank(query, chunks, top_k=3)` re-scores candidates using `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB, cached after first run). Returns top K sorted by `rerank_score` (higher = more relevant).

`generate.py` now fetches 10 candidates from vector search, reranks to 3, then passes to Claude.

#### How the cross-encoder works

**Bi-encoder (vector search):** encodes query and chunk *independently*, then compares vectors:
```
query  → [encoder] → vector A
chunk  → [encoder] → vector B
similarity = cosine_distance(A, B)
```
Fast, but each side has no knowledge of the other. Finds topically similar chunks, not necessarily the best answer.

**Cross-encoder (reranker):** takes query and chunk *together* as one input:
```
[query + chunk] → [encoder] → relevance score (e.g. 0.94)
```
Much more accurate — the model can reason about the relationship between the two texts. But too slow to run against the entire DB (must process every chunk at query time), so used only as a second-pass filter over the top-N candidates.

**Example:**

| Chunk | Bi-encoder distance | Cross-encoder score |
|-------|-------------------|-------------------|
| "Ireland will see 1°C–1.6°C rise by mid-century" | 0.41 | 0.94 |
| "Temperature changes affect agriculture and health" | 0.38 | 0.21 |

The bi-encoder might rank the second chunk higher (similar words). The cross-encoder correctly identifies the first as the actual answer.

---

## Completed Steps (continued)

### Step 8 — Conversation Memory (`chat.py`) ✓

**Problem:** Each call to `ask()` is stateless. You can't ask follow-up questions like "What about the economic impacts?" because the system has no memory of the previous exchange.

**Solution:** `chat.py` is an interactive REPL that maintains conversation history across turns.

**How it works:**
- **Fresh retrieval every turn** — context in the system prompt updates each question, so a follow-up about a different topic still finds the right chunks
- **History in messages** — past turns go in the `messages` array; Claude uses this for conversational context (understanding "that", "this", "tell me more")
- **`MAX_HISTORY_TURNS = 6`** — drops older turns to avoid filling the context window in long conversations

**Design:** The system prompt holds the *retrieved context* (changes every turn). The *messages array* holds the *conversation history* (accumulates across turns). These are two separate concerns — retrieval stays fresh while memory persists.

**Commands:**
```bash
python chat.py       # start a session
> clear              # reset conversation history
> quit               # exit
```

### Step 9 — Evaluation (`evaluate.py`, `eval_data.py`) ✓

**Problem:** How do you know if your RAG system is actually good? Chunk size, overlap, number of results, and reranking all affect quality — but it's hard to tell by eye.

**Solution:** Two-part evaluation harness: retrieval quality (recall@k) and answer quality (LLM-as-judge).

#### Files
- `eval_data.py` — 5 ground-truth question/answer pairs with expected keywords per question
- `evaluate.py` — runs both evaluations and prints a report

#### Retrieval eval (recall@k)
For each question, checks whether the top-k reranked chunks contain all expected keywords. Keyword matching is a practical proxy for "did we retrieve the right chunk?" without having to manually label chunk indices.

#### Answer eval (LLM-as-judge)
Generates an answer for each question, then asks Claude to judge whether it matches the expected answer. Returns YES/NO + a brief reason. More flexible than exact string matching — handles paraphrasing and partial answers correctly.

#### Results on WP786.pdf
```
Recall@3 : 4/5 = 80%
Accuracy : 4/5 = 80%
```

#### Failure analysis
The failing question was: *"What percentage of Ireland's population lives within 10km of the coast?"*

The chunk containing "60% of the population lives within 10km of the coast" was not returned by vector search at all — it didn't appear in the top 10 candidates, so the reranker never saw it. This is a **retrieval failure**, not a reranking failure.

**Root cause:** the 60% fact is buried in a paragraph about other things, diluting its embedding signal. The query vector for "percentage of population within 10km of coast" isn't close enough to that chunk's vector.

**This illustrates the three main RAG failure modes:**
1. **Retrieval failure** — the right chunk isn't fetched (vector search misses it)
2. **Reranking failure** — the right chunk is fetched but ranked too low to be used
3. **Generation failure** — the right chunk is retrieved but Claude gives a wrong answer

**Known fixes:**
- Smaller chunks — so facts aren't diluted by surrounding text
- Larger candidate pool — cast a wider net before reranking (e.g. n=20 instead of 10)
- **Hybrid search** — combine vector search with BM25 (keyword search) so exact matches like "60%" are always surfaced. This is what production RAG systems use and is the first thing frameworks like LangChain/LlamaIndex give you out of the box.

---

## Phase 2 — Framework Migration (LlamaIndex)

Now that you've built every component from scratch and understand how each piece works, we migrate to **LlamaIndex**. The goal is not to replace your understanding — it's to stop maintaining boilerplate and get production-grade features for free.

### Why LlamaIndex over LangChain?

| | LlamaIndex | LangChain |
|--|-----------|-----------|
| Focus | RAG and document Q&A | General LLM chaining |
| API surface | Smaller, more focused | Very large, can be overwhelming |
| RAG primitives | First-class (index, retriever, query engine) | Wrapped in chain abstractions |
| Best for | Document search systems | Multi-step agent workflows |

For a RAG system, LlamaIndex maps more directly to what we've built.

### What each of our files maps to in LlamaIndex

| Our code | LlamaIndex equivalent |
|----------|-----------------------|
| `extract.py` | `SimpleDirectoryReader` |
| `chunk.py` | `SentenceSplitter` / `TokenTextSplitter` |
| `embed.py` | `HuggingFaceEmbedding` |
| `store.py` | `VectorStoreIndex` |
| `search.py` | `VectorIndexRetriever` |
| `rerank.py` | `SentenceTransformerRerank` (postprocessor) |
| `generate.py` | `RetrieverQueryEngine` |
| `chat.py` | `CondenseQuestionChatEngine` |
| `evaluate.py` | `RetrieverEvaluator` + `BatchEvalRunner` |

### What we get for free that we'd have to build ourselves

- **Hybrid search** — BM25 + vector search combined (fixes our 60%/coast failure)
- **Node relationships** — chunks know about their neighbours (better context)
- **Citation / source nodes** — structured source attribution built in
- **Async ingestion** — parallel embedding for large document sets
- **Streaming responses** — token-by-token output to the user
- **Observability** — built-in tracing and callback hooks
- **Many vector store integrations** — Pinecone, Weaviate, pgvector, etc. with one line change

### Migration plan (step by step)

#### Step 10 — Basic index and query engine ✓
**File:** `llama_query.py`
Replaces: `extract.py`, `chunk.py`, `embed.py`, `store.py`, `search.py`, `generate.py`

`Settings.embed_model` and `Settings.llm` are global config — set once, all components use them. `SimpleDirectoryReader` handles PDF parsing. `VectorStoreIndex.from_documents()` does chunking + embedding + storage in one call. `index.as_query_engine()` wraps retrieval + generation.

**Results vs scratch build:**
- Q1 (temperature rise): ✓ answered correctly with more detail than scratch build
- Q2 (emissions targets): ✓ answered correctly
- Q3 (60%/coast): ✗ still failing — same root cause, not yet fixed

**Note:** PDFs moved to `sourcePdfs/` folder. Python upgraded from 3.9 → 3.13 (LlamaIndex requires 3.10+).

#### Step 11 — Persistent storage ✓
**Files:** `llama_index_build.py`, `llama_query.py`

`index.storage_context.persist(persist_dir=INDEX_DIR)` saves the index to disk. `load_index_from_storage()` loads it instantly on subsequent runs — no re-embedding needed. Equivalent to our `./db/` ChromaDB folder.

Also switched to `SentenceSplitter(chunk_size=256, chunk_overlap=32)` — LlamaIndex's built-in chunker that splits on sentence boundaries and measures size in tokens rather than words.

#### Step 12 — Reranking as a postprocessor ✓
`SentenceTransformerRerank` is built into `llama_index.core.postprocessor`. Added as `node_postprocessors=[reranker]` on the query engine — same cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`), same two-stage pattern, one line of config.

#### Step 13 — Hybrid search ✓
**File:** `llama_hybrid.py`

Added `BM25Retriever` alongside the vector retriever, fused with `QueryFusionRetriever` using reciprocal rank fusion (RRF). RRF combines ranked lists from both retrievers by rewarding chunks that rank highly in either list.

**Failure analysis — why Q3 still failed after hybrid search:**
The "60%/coast" chunk was found by BM25, but ranked too low to survive — because the chunk was too large and the specific fact was diluted by surrounding text. The fix was **smaller chunks** (`chunk_size=256`), not hybrid search. This is an important lesson:

| Problem | Fix |
|---------|-----|
| Fact exists but vector search misses it | Hybrid search (BM25 finds exact keywords) |
| Fact is in a large chunk, diluted by other content | Smaller chunks |
| Fact is retrieved but ranked too low | More candidates before reranking |

Always diagnose which failure mode you have before reaching for a solution. Evaluation makes this possible.

After rebuilding the index with smaller chunks, all 3 test questions answered correctly.

#### Step 14 — Chat engine with memory ✓
**File:** `llama_chat.py`

`CondenseQuestionChatEngine` wraps the full hybrid + reranking query engine with conversation memory. On each turn it:
1. Rewrites the user's message + history into a **standalone question** before retrieval
2. Runs that question through the query engine (hybrid retrieval + reranking + generation)
3. Stores the exchange in `chat_history` for the next turn

**Why this is better than our scratch approach:**
Our `chat.py` injected history into the system prompt and sent the raw follow-up to retrieval. A question like "What adaptation measures are proposed for this?" would retrieve poorly because "this" has no meaning as a query.

LlamaIndex rewrites it first:
> *"What adaptation measures are proposed for this?"*
> → *"What adaptation measures are proposed for Ireland in response to its observed temperature increases and climate change trends?"*

Then retrieves. This produces far better results for follow-up questions.

`verbose=True` prints the rewritten query on each turn — useful for understanding and debugging.

**Commands:**
```bash
python llama_chat.py
> clear    # reset conversation history
> quit     # exit
```

#### Step 15 — Re-run evaluation ✓
**File:** `llama_evaluate.py`

Same test set and LLM-as-judge logic as `evaluate.py`, wired to the LlamaIndex hybrid pipeline.

**Final results:**
```
Recall@3 : 5/5 = 100%
Accuracy : 5/5 = 100%
```

**Comparison:**

| | Scratch build | LlamaIndex |
|--|--------------|-----------|
| Recall@3 | 80% | 100% |
| Accuracy | 80% | 100% |

The remaining failure (60%/coast) was fixed by smaller chunks (`chunk_size=256`), not hybrid search. Hybrid search helps when BM25 can surface an exact keyword match that vector search misses — but here the fact was buried in an oversized chunk, diluting both the embedding and the keyword signal. Smaller chunks isolated the fact and made it retrievable.

---

## Quick Reference

```bash
# Setup (first time — requires Python 3.10+, use Homebrew python@3.13 on macOS)
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv venv
source venv/bin/activate
pip install pymupdf sentence-transformers chromadb anthropic \
  llama-index-core llama-index-embeddings-huggingface \
  llama-index-llms-anthropic llama-index-readers-file \
  llama-index-retrievers-bm25

# Every session
source venv/bin/activate
export ANTHROPIC_API_KEY=your-key-here

# ── Scratch pipeline ───────────────────────────────────────
# Ingest a PDF
python store.py sourcePdfs/yourfile.pdf

# Re-ingest from scratch
rm -rf db/ && python store.py sourcePdfs/yourfile.pdf

# Ask a single question
python generate.py "your question here"

# Interactive chat with memory
python chat.py

# Run evaluation
python evaluate.py

# ── LlamaIndex pipeline ────────────────────────────────────
# Build index from all PDFs in sourcePdfs/ (run once, or after adding new PDFs)
rm -rf llama_db/ && python llama_index_build.py

# Ask a single question (hybrid search + reranking)
python llama_hybrid.py

# Interactive chat with memory + question condensing
python llama_chat.py

# Run evaluation
python llama_evaluate.py
```
