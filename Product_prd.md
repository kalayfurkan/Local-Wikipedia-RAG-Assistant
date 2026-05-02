# Product Requirements Document – WikiRAG

## Product Overview

WikiRAG is a locally-running, ChatGPT-style Retrieval-Augmented Generation (RAG) assistant that answers questions about famous people and places using Wikipedia data. The system runs entirely on `localhost` with no external API dependencies.

## Problem Statement

Users need a private, offline-capable AI assistant that can answer factual questions about well-known entities without sending data to cloud services. The system must ground its answers in real data to minimize hallucination.

## Goals

1. Build a fully local RAG system (no cloud APIs)
2. Ingest Wikipedia data for 20+ people and 20+ places
3. Provide accurate, context-grounded answers
4. Deliver a ChatGPT-like user experience via Streamlit
5. Minimize hallucination through strict prompt engineering

## Technical Architecture

```
User Query
    │
    ▼
┌──────────┐     ┌───────────┐     ┌──────────────┐
│  Router   │────▶│ Retriever │────▶│  Generator   │
│(Rule-based)│    │ (ChromaDB) │    │ (Ollama LLM) │
└──────────┘     └───────────┘     └──────────────┘
                       │                    │
                       ▼                    ▼
                 ┌───────────┐       ┌───────────┐
                 │  Embeddings│      │  Response  │
                 │  (nomic)   │      │            │
                 └───────────┘       └───────────┘
```

## Data Pipeline

### Ingest
- Source: Wikipedia REST API (`extracts` module)
- Entities: 20 famous people + 20 famous places
- Output: Clean plaintext stored in SQLite

### Chunk
- Strategy: Sentence-aware chunking with configurable overlap
- Implementation: **Native Python** (no LangChain text splitters)
- Default: 500 chars per chunk, 50 chars overlap

### Embed & Store
- Model: `nomic-embed-text` via Ollama (local)
- Storage: ChromaDB (persistent, single collection)
- Design: **Option B** – single collection with `entity_type` metadata

### Retrieve
- Query embedding via Ollama
- Metadata filtering based on router decision
- Top-K cosine similarity search

### Generate
- Model: `llama3.2` (3B) via Ollama
- Anti-hallucination prompt with strict context-only rules
- Temperature: 0.1 for deterministic outputs

## Technology Stack

| Component     | Technology                    |
|---------------|-------------------------------|
| Language      | Python 3.10+                  |
| LLM           | Ollama (llama3.2 3B)          |
| Embeddings    | nomic-embed-text via Ollama   |
| Vector Store  | ChromaDB (persistent)         |
| Database      | SQLite (stdlib)               |
| UI            | Streamlit                     |
| Frameworks    | ❌ None (native Python)       |
| External APIs | ❌ None (fully local)         |

## Key Design Decisions

### Option B: Single ChromaDB Collection
We chose a single collection with metadata filtering over two separate collections because:
1. **Simpler architecture** – one index to manage
2. **Mixed queries** – "Which famous place is in Turkey?" searches across both types
3. **Scalability** – metadata filtering is O(1) in ChromaDB's HNSW index
4. **Consistency** – single embedding space for better cross-entity comparison

### Native Chunking
Custom sentence-aware chunking with overlap preserves semantic boundaries and provides context continuity between chunks.

### Anti-Hallucination Strategy
- System prompt with explicit "I don't know" fallback
- Low temperature (0.1) for deterministic responses
- Context block clearly delineated in the prompt
- No additional knowledge injection

## Functional Requirements

| ID   | Requirement                                    | Status |
|------|------------------------------------------------|--------|
| FR-1 | Ingest 20+ people from Wikipedia               | ✅      |
| FR-2 | Ingest 20+ places from Wikipedia               | ✅      |
| FR-3 | Chunk documents with overlap                   | ✅      |
| FR-4 | Generate embeddings locally                     | ✅      |
| FR-5 | Store embeddings in vector database             | ✅      |
| FR-6 | Route queries by entity type                    | ✅      |
| FR-7 | Retrieve relevant chunks                        | ✅      |
| FR-8 | Generate grounded answers                       | ✅      |
| FR-9 | Return "I don't know" for unknown queries       | ✅      |
| FR-10| Provide chat-style UI                           | ✅      |
| FR-11| Show source chunks (optional)                   | ✅      |
| FR-12| Clear/reset system                              | ✅      |

## Non-Functional Requirements

- **Latency**: < 30s for answer generation on consumer hardware
- **Privacy**: Zero data leaves the machine
- **Reliability**: Graceful error handling for Ollama outages and missing data files
- **Portability**: Runs on Windows, macOS, Linux
