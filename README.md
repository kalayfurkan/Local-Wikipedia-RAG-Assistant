# WikiRAG – Local Wikipedia RAG Assistant

A fully local, ChatGPT-style Retrieval-Augmented Generation (RAG) system that answers questions about famous people and places using Wikipedia data. **No external APIs, no cloud services** – everything runs on your machine.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                 │
│                   ChatGPT-style Interface                │
└──────────────────────────┬──────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  RAG Pipeline │ (rag_pipeline.py)
                    └──────┬──────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Router   │ │ Retriever│ │ Generator│
        │(rule-based│ │(ChromaDB)│ │(Ollama)  │
        └──────────┘ └────┬─────┘ └────┬─────┘
                          │            │
                    ┌─────▼────┐ ┌─────▼─────┐
                    │ Embeddings│ │ llama3.2  │
                    │  (nomic)  │ │   (3B)    │
                    └──────────┘ └───────────┘
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌──────────┐            ┌──────────┐
        │ ChromaDB │            │  SQLite  │
        │(vectors) │            │(raw text)│
        └──────────┘            └──────────┘
```

## 📦 Components

| File | Description |
|------|-------------|
| `config.py` | Central configuration (DB, Ollama, chunking, entity lists) |
| `ingest.py` | Wikipedia data fetcher + native Python chunker + SQLite writer |
| `embed_and_store.py` | Embedding pipeline (Ollama) + ChromaDB storage |
| `rag_pipeline.py` | Router + Retriever + Generator (full RAG chain) |
| `app.py` | Streamlit chat UI |
| `Product_prd.md` | Product requirements document |
| `recommendation.md` | Production deployment recommendations |

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** (local LLM runtime)

> SQLite is used for raw text storage and ships with Python's standard library — no separate install required.

### 1. Install Ollama

Download and install from [https://ollama.com/download](https://ollama.com/download)

```bash
# Verify installation
ollama --version

# Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text
```

> Ollama runs as a background service on port `11434`. Ensure it is running before proceeding.

### 2. Clone & Install Dependencies

```bash
git clone https://github.com/kalayfurkan/Local-Wikipedia-RAG-Assistant.git
cd Local-Wikipedia-RAG-Assistant

pip install -r requirements.txt
```

### 3. Ingest Wikipedia Data

```bash
python ingest.py
```

This will:
- Fetch Wikipedia articles for 20 people + 20 places
- Clean and chunk the text (sentence-aware, 500 chars, 50 char overlap)
- Store documents and chunks in a local SQLite file (`wiki_rag.db`, auto-created)

> **Note:** If the run reports any `[!] Atlandi: <title>` lines (Wikipedia rate-limiting), simply re-run `python ingest.py`. Already-ingested entities are skipped automatically — only missing ones are fetched, so the second run completes in seconds.

### 4. Embed & Store in ChromaDB

```bash
python embed_and_store.py
```

This will:
- Read all chunks from SQLite
- Generate embeddings using `nomic-embed-text` via Ollama
- Store vectors in ChromaDB with entity type metadata

### 5. Launch the Chat UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🗣️ Example Queries

### People
| Query | Expected Behavior |
|-------|-------------------|
| Who was Albert Einstein and what is he known for? | Grounded answer from Einstein's Wikipedia data |
| What did Marie Curie discover? | Answer about radioactivity from Curie's article |
| Why is Nikola Tesla famous? | Details from Tesla's Wikipedia page |
| Compare Lionel Messi and Cristiano Ronaldo | Side-by-side comparison from both articles |

### Places
| Query | Expected Behavior |
|-------|-------------------|
| Where is the Eiffel Tower located? | Location details from Eiffel Tower article |
| What was the Colosseum used for? | Historical usage from Colosseum article |
| What is Machu Picchu? | Description from Machu Picchu article |

### Mixed
| Query | Expected Behavior |
|-------|-------------------|
| Which famous place is located in Turkey? | Should find Hagia Sophia / Galata Tower |
| Compare Albert Einstein and Nikola Tesla | Cross-person comparison |

### Failure Cases (should return "I don't know")
| Query | Expected Behavior |
|-------|-------------------|
| Who is the president of Mars? | "I don't know based on the available data." |
| Tell me about John Doe | "I don't know based on the available data." |

---

## 🔧 Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between consecutive chunks |
| `OLLAMA_LLM_MODEL` | llama3.2 | LLM model for generation |
| `OLLAMA_EMBED_MODEL` | nomic-embed-text | Embedding model |
| `TOP_K` | 5 | Number of chunks retrieved per query |

---

## 🎯 Design Decisions

### Chunking Strategy
- **Sentence-aware** splitting instead of fixed character cuts
- Preserves semantic boundaries for better retrieval quality
- Overlap ensures context continuity between adjacent chunks
- Implemented in **pure Python** – no LangChain/LlamaIndex

### Vector Store: Option B (Single Collection + Metadata)
- One ChromaDB collection with `entity_type` metadata (`person` / `place`)
- Enables mixed queries without cross-collection merging
- Simpler architecture, easier to maintain

### Anti-Hallucination
- Strict system prompt: "Answer ONLY from context"
- Explicit "I don't know" fallback instruction
- Low temperature (0.1) for deterministic outputs
- Context block clearly delineated in the prompt

---

## 📁 Project Structure

```
Local-Wikipedia-RAG-Assistant/
├── config.py              # Central configuration
├── ingest.py              # Wikipedia ingestion + native chunking
├── embed_and_store.py     # Embedding + ChromaDB storage
├── rag_pipeline.py        # Router + Retriever + Generator
├── app.py                 # Streamlit chat UI
├── requirements.txt       # Python dependencies
├── Product_prd.md         # Product requirements document
├── recommendation.md      # Production deployment guide
├── README.md              # This file
├── wiki_rag.db            # SQLite raw-text store (auto-created)
└── chroma_db/             # ChromaDB persistent storage (auto-created)
```

---

## 📹 Demo Video

**▶ Watch the 5-minute demo on YouTube:** https://www.youtube.com/watch?v=xu9bivuu-ok

The video covers: system overview, live ingestion + Q&A, model and retrieval choices, tradeoffs, and possible improvements.

---

## ⚠️ Technical Constraints

- ✅ Runs fully on `localhost`
- ✅ No external LLM APIs (OpenAI, Anthropic, etc.)
- ✅ No RAG frameworks (LangChain, LlamaIndex)
- ✅ Core chunking logic is native Python
- ✅ Local embedding via Ollama
- ✅ Local LLM via Ollama
