# Production Deployment Recommendations

This document outlines how to evolve the WikiRAG system from a local development prototype into a production-grade deployment.

---

## 1. Infrastructure & Deployment

### Containerization (Docker)

The entire stack should be containerized for reproducibility and portability:

```yaml
# docker-compose.yml (conceptual)
services:
  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes: [ollama_data:/root/.ollama]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  app:
    build: .
    ports: ["8501:8501"]
    depends_on: [ollama, postgres, chroma]
    volumes: [app_data:/app/data]   # SQLite file (dev) → Postgres (prod)

  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: wiki_rag

  chroma:
    image: chromadb/chroma:latest
    ports: ["8000:8000"]
    volumes: [chroma_data:/chroma/chroma]
```

> The local prototype uses **SQLite** (single-file, zero-setup). For production, swap to **PostgreSQL** for concurrent writes, replication, and managed-service availability.

**Benefits:**
- Reproducible builds across environments
- Easy horizontal scaling
- GPU passthrough for Ollama

### Kubernetes (K8s)

For larger scale:
- Deploy Ollama as a StatefulSet with GPU node affinity
- ChromaDB and PostgreSQL as persistent StatefulSets
- Streamlit frontend as a Deployment with HPA (Horizontal Pod Autoscaler)
- Use Ingress controllers for HTTPS termination

---

## 2. Model Serving

### Current Limitation
- Ollama is single-tenant, sequential inference
- No request queuing or load balancing

### Production Recommendations

| Approach | Use Case | Notes |
|----------|----------|-------|
| **vLLM** | High-throughput serving | Supports batching, PagedAttention, OpenAI-compatible API |
| **TGI (Text Generation Inference)** | Hugging Face ecosystem | Optimized for transformer models |
| **Triton Inference Server** | Multi-model serving | NVIDIA-backed, supports TensorRT-LLM |
| **Ollama + Load Balancer** | Simple scaling | Multiple Ollama instances behind nginx/HAProxy |

### GPU Considerations
- Minimum: 1× NVIDIA T4 (16 GB VRAM) for 7B models
- Recommended: 1× A10G (24 GB) or A100 (40/80 GB)
- For 3B models (llama3.2): Can run on consumer GPUs (RTX 3060+)

---

## 3. Vector Store Scaling

### Current: ChromaDB (Embedded)
- Single-process, file-based
- Good for < 100K vectors
- No built-in replication

### Production Options

| Solution | Max Scale | Features |
|----------|-----------|----------|
| **ChromaDB Server** | ~1M vectors | Client-server mode, REST API |
| **Pinecone** | Billions | Managed, auto-scaling, serverless |
| **Weaviate** | Billions | Self-hosted, multi-modal, GraphQL |
| **Qdrant** | Billions | High performance, filtering, on-prem |
| **Milvus** | Billions | Distributed, K8s native, GPU indexing |
| **pgvector** | ~10M | PostgreSQL extension, familiar SQL |

### Recommendation
For a self-hosted production system, **Qdrant** or **Milvus** provide the best balance of performance, filtering capabilities, and operational maturity. If cloud-managed is acceptable, **Pinecone** minimizes operational overhead.

---

## 4. Database

### Current: SQLite (Single File)
- Zero-setup, embedded in the Python process
- Great for prototyping; not for concurrent writes or high availability
- Single point of failure (one file)

### Production Enhancements
1. **Migrate to PostgreSQL**: Concurrent writes, replication, managed services (RDS / Cloud SQL)
2. **Connection Pooling**: `SQLAlchemy` or `psycopg_pool` (pool_size=10, max_overflow=20)
3. **Read Replicas**: PostgreSQL streaming replication for read scaling
4. **Consolidation Option**: PostgreSQL + `pgvector` to merge raw-text and vector storage in one engine
5. **Backups**: Point-in-time recovery via WAL archiving (managed services do this automatically)

---

## 5. Ingestion Pipeline

### Current Limitations
- Sequential, single-threaded Wikipedia fetching
- No incremental updates
- No scheduling

### Production Architecture

```
┌────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Scheduler │────▶│  Fetcher  │────▶│  Chunker  │────▶│  Embedder │
│  (Airflow)  │    │ (async)   │    │ (parallel) │    │  (batch)   │
└────────────┘     └──────────┘     └──────────┘     └──────────┘
```

- **Apache Airflow** or **Celery Beat** for scheduled re-ingestion
- **Async HTTP** (aiohttp) for parallel Wikipedia fetching
- **Batch embedding** with GPU acceleration
- **Incremental updates**: Hash-based change detection, only re-embed modified documents
- **Dead Letter Queue**: Failed ingestions routed to DLQ for retry

---

## 6. Retrieval Improvements

### Hybrid Search
Combine vector similarity with keyword search (BM25):
```
Final Score = α × vector_score + (1 - α) × bm25_score
```

### Re-Ranking
Add a cross-encoder re-ranker after initial retrieval:
1. Retrieve top-50 by vector similarity
2. Re-rank with a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
3. Return top-5 re-ranked results

### Query Expansion
- Use the LLM to rephrase/expand the query before retrieval
- Generate multiple query variants and merge results (reciprocal rank fusion)

### Metadata-Enhanced Filtering
- Add date ranges, categories, and popularity scores to metadata
- Enable faceted search in the UI

---

## 7. Security

### Authentication & Authorization
- Add user authentication (OAuth2, JWT tokens)
- Rate limiting per user (e.g., 10 requests/minute)
- API key management for programmatic access

### Data Security
- Encrypt the relational store at rest (TDE / `pgcrypto`) and in transit (TLS)
- ChromaDB behind a private network (no public access)
- Ollama exposed only to the application tier

### Input Sanitization
- Validate and sanitize all user inputs
- Prevent prompt injection attacks
- Log and monitor suspicious queries

---

## 8. Observability

### Logging
- Structured logging (JSON) with `structlog` or `loguru`
- Log every pipeline stage: route → retrieve → generate
- Include latency metrics per stage

### Monitoring
- **Prometheus + Grafana** for metrics dashboards
- Track: query latency, retrieval quality, LLM token usage
- Alert on: Ollama downtime, database connection failures, embedding errors

### Tracing
- **OpenTelemetry** for distributed tracing across pipeline stages
- Trace ID propagation from UI → router → retriever → generator

---

## 9. Caching

### Response Caching
- Cache frequent query results in **Redis**
- TTL-based expiration (e.g., 1 hour)
- Cache key: normalized query hash

### Embedding Caching
- Cache query embeddings to avoid re-computation
- Store in Redis with query text as key

### Estimated Impact
- 60-80% reduction in LLM calls for repeated questions
- Sub-100ms response for cached queries

---

## 10. UI / UX Improvements

### Current: Streamlit
- Great for prototyping, limited for production

### Production Frontend
- **Next.js / React** with SSR for SEO and performance
- WebSocket connection for real-time streaming responses
- Mobile-responsive design
- Accessibility (WCAG 2.1 compliance)

### Features to Add
- User accounts and chat history persistence
- Response feedback (thumbs up/down) for quality monitoring
- Citation highlighting (click source to see original Wikipedia section)
- Multi-language support
- Dark/light theme toggle

---

## 11. Cost Estimation (Self-Hosted)

| Component | Monthly Cost (AWS) | Notes |
|-----------|-------------------|-------|
| GPU Instance (g4dn.xlarge) | ~$380 | T4 GPU, for Ollama |
| PostgreSQL (db.t3.medium) | ~$50 | RDS, Multi-AZ |
| ChromaDB (t3.large) | ~$60 | EC2 + EBS |
| Load Balancer (ALB) | ~$25 | HTTPS termination |
| **Total** | **~$515/mo** | For moderate traffic |

---

## 12. Migration Path

### Phase 1: Stabilize (Week 1-2)
- Dockerize all components
- Add health checks and basic monitoring
- Implement connection pooling
- Add response caching (Redis)

### Phase 2: Scale (Week 3-4)
- Migrate to ChromaDB server mode or Qdrant
- Deploy Ollama with vLLM for better throughput
- Set up CI/CD pipeline
- Add authentication

### Phase 3: Optimize (Week 5-8)
- Implement hybrid search (vector + BM25)
- Add re-ranking with cross-encoder
- Build production frontend (React/Next.js)
- Set up full observability stack

### Phase 4: Operate (Ongoing)
- Scheduled data re-ingestion (Airflow)
- A/B testing for retrieval strategies
- Model upgrades (larger models as hardware allows)
- User feedback loop for quality improvement

---

## Summary

The current WikiRAG prototype demonstrates a sound architecture. The key production evolution areas are:

1. **Containerization** → Docker + Kubernetes
2. **Model serving** → vLLM or TGI for throughput
3. **Vector store** → Qdrant or Milvus for scale
4. **Caching** → Redis for latency reduction
5. **Observability** → Prometheus + Grafana + OpenTelemetry
6. **Security** → Auth, rate limiting, encryption
7. **Frontend** → React/Next.js for production UX

The system's modular design (separate ingest, embed, retrieve, generate) makes this evolution straightforward – each component can be upgraded independently without a full rewrite.
