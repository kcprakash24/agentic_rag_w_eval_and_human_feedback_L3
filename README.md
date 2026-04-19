# Agentic RAG: Reranking, Self-RAG, Evals & Human Feedback

A research paper Q&A system built with LangGraph, pgvector, Docling, and Langfuse.  
Covers layout-aware PDF ingestion, two-stage retrieval, Self-RAG grading, automated evaluation, and human feedback collection.


## What This Builds

- Parses complex PDFs using Docling's layout-aware pipeline
- Routes questions to the correct document collection
- Retrieves top-20 chunks from pgvector, reranks to top-4 with a cross-encoder
- Caches answers in Redis; invalidates cache on negative feedback
- Runs automated BLEU + ROUGE-L evals against a 20-question golden dataset
- Logs human 👍/👎 feedback to Langfuse and Postgres
- Displays eval trends and feedback history in a Streamlit admin panel

## Notes
- The full value of human feedback as a training signal (RLHF-style) requires thousands of rated examples to be statistically meaningful. With a small user base, treat feedback as a qualitative signal for human review — not as an automatic reranking input.
- Self-RAG graders were built and tested (Step 5) but dropped from the final agent to reduce latency. The code exists in `graders.py` and can be re-enabled in `nodes.py`.
- The reranker model runs locally — first call downloads ~80MB. Subsequent calls are fast.
- Ollama must be running before starting the UI or any notebook cells that invoke the agent.
- Redis and Postgres must be running via Docker before any agent calls.

> **Chunking Token Selection Criteria:** Rule of thumb: 1 token ≈ 4 chars ≈ 0.75 words. So 512 tokens ≈ 384 words ≈ 2048 chars. For this project, avg_chars of 1033 ≈ 258 words ≈ 344 tokens — well within the 512 limit.

## Tech Stack

| Component | Choice |
|---|---|
| PDF Parser | Docling |
| Chunking | Docling HybridChunker |
| Vector Store | pgvector (multi-collection + HNSW index) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| LLM | gemma4 via Ollama (local) |
| Evaluation | RAGAS non-LLM metrics (BleuScore + RougeScore) |
| Observability | Langfuse (traces, scores, datasets) |
| Cache | Redis semantic cache |
| Memory | Postgres (chat history + summarization) |
| Agent Framework | LangGraph |
| UI | Streamlit |

---

## Folder Structure

```
agentic_rag_l3/
├── .env
├── .env.example
├── .gitignore
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
│
├── data/
│   └── papers/                     # PDF files to ingest
│
├── notebooks/
│   └── agentic_rag.ipynb           # unified notebook across all steps
│
├── agentic_rag/
│   ├── config.py
│   │
│   ├── ingestion/
│   │   ├── loader.py               # Docling DocumentConverter
│   │   ├── chunker.py              # HybridChunker
│   │   └── collection_router.py    # filename → collection name
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── llm/
│   │   └── provider.py
│   │
│   ├── vectorstore/
│   │   └── pgvector_store.py       # multi-collection + HNSW + feedback write
│   │
│   ├── retrieval/
│   │   ├── retriever.py            # top-20 from pgvector
│   │   └── reranker.py             # cross-encoder → top-4
│   │
│   ├── memory/
│   │   ├── pg_memory.py
│   │   └── summarizer.py
│   │
│   ├── cache/
│   │   └── redis_cache.py
│   │
│   ├── agent/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── graph.py
│   │   └── graders.py              # context + answer graders (Self-RAG)
│   │
│   ├── evaluation/
│   │   ├── dataset.py              # 20 golden Q&A pairs → Langfuse dataset
│   │   ├── scorer.py               # BleuScore + RougeScore (non-LLM)
│   │   └── run_evals.py            # batch runner → eval_results table
│   │
│   └── observability/
│       └── langfuse_client.py      # tracing + feedback score submission
│
└── ui/
    └── app.py                      # Streamlit chat UI + admin panel
```

---

## Agent Graph

![Agentic Graph](graph.png)

## Prerequisites

- [Ollama](https://ollama.com) running locally with `gemma4:e2b` pulled: `ollama pull gemma4:e2b`
- Docker (for Postgres + Redis)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)


## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/kcprakash24/agentic_rag_L3.git
cd agentic_rag_L3
uv sync
```

**2. Copy and fill environment variables**

```bash
cp .env.example .env
```

Edit `.env`:

```
POSTGRES_URL=postgresql://docmind:docmind@localhost:5432/docmind
REDIS_URL=redis://localhost:6379
OLLAMA_BASE_URL=http://localhost:11434
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

**3. Start Postgres + Redis**

```bash
docker-compose up -d
```

**4. Run schema migrations**

```bash
docker exec -i <container_name> psql -U <user_name> -d <user_name> < migrations/schema.sql
```

**5. Ingest papers**

Put your PDFs in `data/papers/`, then run the ingestion cells in the notebook.


## Database Schema

```sql
-- Documents (multi-collection)
CREATE TABLE documents (
    id         SERIAL PRIMARY KEY,
    chunk_id   TEXT UNIQUE NOT NULL,
    content    TEXT NOT NULL,
    metadata   JSONB,
    embedding  vector(768),
    collection TEXT DEFAULT 'general'
);
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Chat history
CREATE TABLE chat_messages (
    id         SERIAL PRIMARY KEY,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Compressed summaries
CREATE TABLE chat_summaries (
    id            SERIAL PRIMARY KEY,
    user_id       TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    summary       TEXT NOT NULL,
    message_count INT NOT NULL,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- Human feedback
CREATE TABLE user_feedback (
    id         SERIAL PRIMARY KEY,
    trace_id   TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    thumbs_up  BOOLEAN NOT NULL,
    comment    TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Eval results
CREATE TABLE eval_results (
    id          SERIAL PRIMARY KEY,
    trace_id    TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    reference   TEXT NOT NULL,
    bleu_score  FLOAT,
    rouge_score FLOAT,
    dataset_run TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

**6. Run the UI**

```bash
streamlit run ui/app.py
```

## Evaluation

Builds an offline eval pipeline using RAGAS non-LLM metrics — no judge model required.

**Metrics used:**

| Metric | What it measures | Needs ground truth |
|---|---|---|
| BleuScore | n-gram precision of answer vs reference | Yes |
| RougeScore (ROUGE-L F1) | longest common subsequence overlap | Yes |

**Golden dataset:** 20 hand-written Q&A pairs from the "Attention Is All You Need" paper, stored as a Langfuse dataset (`attention_paper_golden_v1`).

**Run evals:**

```bash
python -m agentic_rag.evaluation.dataset    # upload golden pairs (once)
python -m agentic_rag.evaluation.run_evals  # run all 20 questions + log scores
```

Scores are logged to Langfuse per trace and written to the `eval_results` table. The Streamlit admin panel shows score trends across runs.

**Interpreting scores:**

BLEU and ROUGE are relative metrics — their value is detecting regressions between runs, not claiming absolute quality. A drop of 0.05+ in avg ROUGE-L after a code change is a signal worth investigating.

---

## Human Feedback Loop

After every answer, the UI shows 👍/👎 buttons and an optional comment box.

On submission:
- Score logged to Langfuse on the originating trace (`human_feedback`: 1.0 or 0.0)
- Row written to `user_feedback` table in Postgres

The admin panel (sidebar toggle) shows:
- Avg BLEU and ROUGE-L across all eval runs
- Score trend line chart per eval run
- Recent 10 feedback rows with thumbs and comments

**Closing the eval loop:** Thumbs-down responses with useful comments can be promoted to new golden dataset items, growing the eval set from real failures over time.

