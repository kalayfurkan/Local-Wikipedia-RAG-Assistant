# ============================================================
# rag_pipeline.py  –  Router + Retriever + Generator
# ============================================================
#  • Kural tabanlı router: sorgunun person/place/both olduğunu belirler
#  • ChromaDB'den metadata filtrelemesiyle chunk retrieve eder
#  • Ollama LLM ile bağlama dayalı cevap üretir (halüsinasyon korumalı)
#  • Hazır framework KULLANILMAZ – tüm logic native Python
# ============================================================

import os
import re
import sqlite3
import requests
import chromadb
from config import (
    OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL,
    SQLITE_DB_PATH, PEOPLE, PLACES,
)

# ── ChromaDB ayarları ────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "wiki_rag"
TOP_K = 5            # Üretici'ye verilecek nihai chunk sayısı
HYBRID_POOL_SIZE = 20   # Dense ve BM25'ten alınan ham aday havuzu (her biri)
RRF_K = 60           # Reciprocal Rank Fusion sabiti (literatür default)


# ════════════════════════════════════════════════════════════
#  1.  KURAL TABANLI ROUTER
# ════════════════════════════════════════════════════════════

# ── Yer anahtar kelimeleri (lowercase) ───────────────────────
PLACE_KEYWORDS = {
    # Genel yer terimleri
    "tower", "wall", "canyon", "mountain", "mount", "falls",
    "pyramid", "statue", "reef", "temple", "mosque", "church",
    "cathedral", "palace", "castle", "bridge", "monument",
    "landmark", "building", "structure", "ruin", "ruins",
    "island", "lake", "river", "ocean", "sea", "volcano",
    # Soru kalıpları
    "located", "location", "where", "built", "visit",
    "country", "city", "height", "tall", "architecture",
}

# ── Kişi anahtar kelimeleri (lowercase) ──────────────────────
PERSON_KEYWORDS = {
    # Genel kişi terimleri
    "who", "born", "died", "life", "biography", "career",
    "scientist", "artist", "musician", "writer", "poet",
    "inventor", "physicist", "chemist", "player", "footballer",
    "singer", "painter", "philosopher", "emperor", "queen",
    "king", "president", "leader", "Nobel", "award",
    "discover", "discovered", "discovery", "invention",
    "famous for", "known for", "achievement",
}

# ── Karşılaştırma anahtar kelimeleri ─────────────────────────
COMPARE_KEYWORDS = {"compare", "comparison", "difference", "vs", "versus",
                    "similarities", "similar", "between"}


def _build_entity_lookup() -> tuple[dict[str, str], dict[str, str]]:
    """
    Config'deki varlık listelerinden hızlı lookup sözlükleri oluşturur.
    { "albert einstein": "person", "eiffel tower": "place", ... }
    Ayrıca her kelimeyi ayrı ayrı da eşler (kısmi eşleşme için).
    """
    entity_map: dict[str, str] = {}   # tam isim → tip
    word_map: dict[str, str] = {}     # tek kelime → tip

    for name in PEOPLE:
        entity_map[name.lower()] = "person"
        for word in name.lower().split():
            if len(word) > 2:  # "da", "jr" gibi kısa kelimeleri atla
                word_map[word] = "person"

    for name in PLACES:
        entity_map[name.lower()] = "place"
        for word in name.lower().split():
            if len(word) > 2 and word not in {"the", "of"}:
                word_map[word] = "place"

    return entity_map, word_map


ENTITY_MAP, WORD_MAP = _build_entity_lookup()

# Lowercase isim → orijinal başlık (ChromaDB metadatasındaki casing)
TITLE_LOOKUP = {n.lower(): n for n in PEOPLE + PLACES}


def match_entity_titles(query: str) -> list[str]:
    """
    Sorguda tam adı geçen tüm bilinen varlıkların orijinal başlıklarını döner.
    "Compare Albert Einstein and Nikola Tesla" → ["Albert Einstein", "Nikola Tesla"]
    "What did she discover?" → []
    """
    q_lower = query.lower()
    return [TITLE_LOOKUP[name] for name in ENTITY_MAP if name in q_lower]


def route_query(query: str) -> str:
    """
    Kullanıcı sorgusunun tipini belirler.

    Dönüş değerleri:
        "person"  → Sorgu bir kişiyle ilgili
        "place"   → Sorgu bir yerle ilgili
        "both"    → Sorgu hem kişi hem yer içeriyor veya belirsiz

    Algoritma:
    ---------
    1. Sorguda bilinen varlık isimlerini ara (tam eşleşme)
    2. Karşılaştırma sorgusu mu kontrol et
    3. Anahtar kelimelere göre puanla
    4. En yüksek puanlı kategoriyi döndür
    """
    q_lower = query.lower()
    words = set(q_lower.split())

    # ── 1. Tam varlık ismi eşleşmesi ────────────────────────
    found_types = set()
    for entity_name, entity_type in ENTITY_MAP.items():
        if entity_name in q_lower:
            found_types.add(entity_type)

    # İki farklı tip bulunduysa → "both"
    if len(found_types) == 2:
        return "both"

    # ── 2. Karşılaştırma sorgusu kontrolü ────────────────────
    is_compare = bool(words & COMPARE_KEYWORDS)

    # Karşılaştırma + farklı tipler → both
    if is_compare and len(found_types) >= 1:
        # Kelime bazlı ikinci varlık ara
        word_types = set()
        for word in words:
            if word in WORD_MAP:
                word_types.add(WORD_MAP[word])
        if len(word_types | found_types) >= 2:
            return "both"

    # Tek tip tam eşleşme bulunduysa direkt döndür
    if len(found_types) == 1:
        return found_types.pop()

    # ── 3. Kelime bazlı varlık eşleşmesi ────────────────────
    word_types = set()
    for word in words:
        if word in WORD_MAP:
            word_types.add(WORD_MAP[word])

    if len(word_types) == 2:
        return "both"
    if len(word_types) == 1:
        return word_types.pop()

    # ── 4. Anahtar kelime puanlama ───────────────────────────
    person_score = len(words & PERSON_KEYWORDS)
    place_score = len(words & PLACE_KEYWORDS)

    if person_score > 0 and place_score > 0:
        return "both"
    if person_score > place_score:
        return "person"
    if place_score > person_score:
        return "place"

    # ── 5. Belirsiz → her ikisinden de ara ───────────────────
    return "both"


# ════════════════════════════════════════════════════════════
#  2.  RETRIEVER  (ChromaDB + metadata filtreleme)
# ════════════════════════════════════════════════════════════

def _get_collection() -> chromadb.Collection:
    """Mevcut ChromaDB koleksiyonunu döner."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_collection(name=COLLECTION_NAME)


def _embed_query(text: str) -> list[float]:
    """
    Sorguyu Ollama nomic-embed-text ile embed eder.
    Native HTTP isteği – framework yok.
    """
    url = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {"model": OLLAMA_EMBED_MODEL, "input": [text]}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


# ── BM25 (SQLite FTS5) için stopwords ve tokenizer ───────────
_STOPWORDS = {
    "a","an","the","of","in","on","at","to","for","from","by","with","about",
    "into","over","under","between","through",
    "is","are","was","were","be","been","being","am",
    "i","me","my","we","us","our","you","your","he","him","his","she","her",
    "it","its","they","them","their",
    "and","or","but","not","no","so","if","as","that","this","these","those",
    "what","who","whom","when","where","why","how","which","whose",
    "do","does","did","done","have","has","had","can","could","will","would",
    "should","shall","may","might","must",
    "tell","say","said","get","got","make","made","give","gave","know",
    "all","any","some","many","much","few","more","most","other","another",
    "very","just","also","than","then","there","here",
}


def _tokenize_for_bm25(query: str) -> list[str]:
    """Sorguyu BM25 için içerik kelimelerine ayır (alphanumeric, ≥3 char)."""
    tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]


def bm25_search(query: str, query_type: str | None = None,
                matched_titles: list[str] | None = None,
                top_k: int = HYBRID_POOL_SIZE) -> list[dict]:
    """
    SQLite FTS5 üzerinde BM25 ranking ile keyword araması.
    Aynı metadata filtreleri (entity_type, matched titles) burada da uygulanır.
    Eşleşme yoksa boş liste döner — hybrid füzyon dense'e geri düşer.
    """
    tokens = _tokenize_for_bm25(query)
    if not tokens:
        return []

    # OR'lu sorgu: tek bir nadir keyword (e.g. "turkey") bile eşleşmeye yeter
    fts_match = " OR ".join(tokens)

    sql = [
        "SELECT c.id, c.title, c.entity_type, c.chunk_index, c.chunk_text, "
        "       bm25(chunks_fts) AS score",
        "FROM chunks_fts",
        "JOIN chunks c ON c.id = chunks_fts.rowid",
        "WHERE chunks_fts MATCH ?",
    ]
    params: list = [fts_match]

    if query_type in ("person", "place"):
        sql.append("AND c.entity_type = ?")
        params.append(query_type)

    if matched_titles:
        placeholders = ",".join("?" * len(matched_titles))
        sql.append(f"AND c.title IN ({placeholders})")
        params.extend(matched_titles)

    sql.append("ORDER BY score LIMIT ?")
    params.append(top_k)

    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        rows = conn.execute(" ".join(sql), params).fetchall()
    except sqlite3.OperationalError as exc:
        print(f"  [WARN] BM25 sorgusu basarisiz, dense-only: {exc}")
        return []
    finally:
        conn.close()

    return [
        {
            "chunk_id": r[0],
            "title": r[1],
            "entity_type": r[2],
            "chunk_index": r[3],
            "text": r[4],
            "bm25_score": r[5],
        }
        for r in rows
    ]


def _rrf_combine(dense: list[dict], bm25_hits: list[dict],
                 top_k: int = TOP_K, k: int = RRF_K) -> list[dict]:
    """
    Reciprocal Rank Fusion: aynı chunk iki listede varsa RRF puanları
    toplanır.  k=60 literatür default'u, az sayıda sonuç için iyi çalışır.
    """
    pool: dict[int, dict] = {}

    for rank, item in enumerate(dense, start=1):
        cid = item["chunk_id"]
        pool[cid] = {"item": item, "rrf": 1.0 / (k + rank)}

    for rank, item in enumerate(bm25_hits, start=1):
        cid = item["chunk_id"]
        if cid in pool:
            pool[cid]["rrf"] += 1.0 / (k + rank)
        else:
            pool[cid] = {"item": item, "rrf": 1.0 / (k + rank)}

    ranked = sorted(pool.values(), key=lambda x: -x["rrf"])
    return [entry["item"] for entry in ranked[:top_k]]


def retrieve(query: str, query_type: str | None = None,
             top_k: int = TOP_K) -> list[dict]:
    """
    Hybrid retrieval: dense (Chroma) + BM25 (SQLite FTS5) → RRF fusion.

    Akış
    ----
    1. Router → query_type ve matched_titles
    2. Dense: Chroma'dan HYBRID_POOL_SIZE chunk
    3. BM25: SQLite FTS5'ten HYBRID_POOL_SIZE chunk (aynı metadata filtreleri)
    4. RRF fusion → top_k

    Dönüş: [{chunk_id, text, title, entity_type, chunk_index,
             distance?, bm25_score?}, ...]
    """
    if query_type is None:
        query_type = route_query(query)

    matched_titles = match_entity_titles(query)

    # ── Metadata filtreleri (her iki retrieval'da paylaşılır) ─
    chroma_clauses = []
    if query_type in ("person", "place"):
        chroma_clauses.append({"entity_type": query_type})
    if matched_titles:
        if len(matched_titles) == 1:
            chroma_clauses.append({"title": matched_titles[0]})
        else:
            chroma_clauses.append({"title": {"$in": matched_titles}})

    if not chroma_clauses:
        where_filter = None
    elif len(chroma_clauses) == 1:
        where_filter = chroma_clauses[0]
    else:
        where_filter = {"$and": chroma_clauses}

    # ── 1) Dense (Chroma) ────────────────────────────────────
    collection = _get_collection()
    query_embedding = _embed_query(query)

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": HYBRID_POOL_SIZE,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    raw = collection.query(**kwargs)

    dense_results = []
    for cid, doc, meta, dist in zip(
        raw["ids"][0], raw["documents"][0],
        raw["metadatas"][0], raw["distances"][0],
    ):
        # Chroma id formati: "chunk_<sqlite_id>"
        chunk_id = int(cid.replace("chunk_", "")) if cid.startswith("chunk_") else cid
        dense_results.append({
            "chunk_id": chunk_id,
            "text": doc,
            "title": meta["title"],
            "entity_type": meta["entity_type"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        })

    # ── 2) BM25 (SQLite FTS5) ────────────────────────────────
    bm25_results = bm25_search(
        query,
        query_type=query_type,
        matched_titles=matched_titles or None,
        top_k=HYBRID_POOL_SIZE,
    )

    # ── 3) RRF fusion ────────────────────────────────────────
    fused = _rrf_combine(dense_results, bm25_results, top_k=top_k, k=RRF_K)

    # Sadece BM25'ten gelen item'lar 'distance' içermez — UI/API tutarlılığı için NaN backfill
    for item in fused:
        item.setdefault("distance", float("nan"))
        item.setdefault("bm25_score", float("nan"))

    return fused


# ════════════════════════════════════════════════════════════
#  3.  GENERATOR  (Ollama LLM – halüsinasyon korumalı)
# ════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful assistant that answers questions ONLY based on the provided context.

STRICT RULES:
1. Use ONLY the information in the CONTEXT below to answer the question.
2. If the context does not contain enough information to answer the question, respond EXACTLY with: "I don't know based on the available data."
3. Do NOT make up facts, speculate, or use any knowledge outside the provided context.
4. If the question asks about a topic not covered in the context, respond with: "I don't know based on the available data."
5. Keep your answers concise, accurate, and well-structured.
6. When comparing entities, use only facts from the context for each entity.
7. If asked about something completely unrelated to the context (e.g., "president of Mars"), respond with: "I don't know based on the available data."
"""


def _build_context_block(chunks: list[dict]) -> str:
    """
    Retrieve edilen chunk'lardan LLM'e verilecek context bloğunu oluşturur.
    """
    if not chunks:
        return "No relevant context found."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[Source {i}: {chunk['title']} ({chunk['entity_type']})]\n"
            f"{chunk['text']}\n"
        )
    return "\n".join(lines)


def generate(query: str, context_chunks: list[dict],
             stream: bool = False) -> str | None:
    """
    Ollama LLM'e bağlamla birlikte soru gönderir ve cevap üretir.

    Parametreler
    ------------
    query          : Kullanıcı sorusu
    context_chunks : retrieve() fonksiyonundan dönen chunk listesi
    stream         : True ise streaming yanıt döner (generator)

    Dönüş
    ------
    str – LLM yanıtı
    """
    context_block = _build_context_block(context_chunks)

    user_message = (
        f"CONTEXT:\n"
        f"─────────────────────────────────\n"
        f"{context_block}\n"
        f"─────────────────────────────────\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer the question using ONLY the context above. "
        f"If the answer is not in the context, say exactly: "
        f"\"I don't know based on the available data.\""
    )

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": stream,
        "options": {
            "temperature": 0.1,      # Düşük sıcaklık → daha deterministik
            "top_p": 0.9,
            "num_predict": 1024,     # Max token
        },
    }

    try:
        if not stream:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        else:
            # Streaming mode: satır satır JSON döner
            resp = requests.post(url, json=payload, timeout=120, stream=True)
            resp.raise_for_status()
            full_response = ""
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    import json
                    chunk_data = json.loads(line)
                    token = chunk_data.get("message", {}).get("content", "")
                    full_response += token
                    print(token, end="", flush=True)
                    if chunk_data.get("done", False):
                        break
            print()  # Yeni satır
            return full_response

    except requests.RequestException as exc:
        print(f"  [ERROR] Ollama LLM isteği başarısız: {exc}")
        return None


# ════════════════════════════════════════════════════════════
#  4.  TAM RAG PIPELINE  (Router → Retrieve → Generate)
# ════════════════════════════════════════════════════════════

def ask(query: str, top_k: int = TOP_K,
        stream: bool = False,
        show_sources: bool = False) -> dict:
    """
    Tam RAG pipeline'ı çalıştırır.

    Parametreler
    ------------
    query        : Kullanıcı sorusu
    top_k        : Retrieve edilecek chunk sayısı
    stream       : Streaming yanıt
    show_sources : True ise kaynak chunk'ları da döner

    Dönüş
    ------
    {
        "query": str,
        "query_type": "person" | "place" | "both",
        "answer": str,
        "sources": [{"title": ..., "text": ..., ...}, ...]  (opsiyonel)
    }
    """
    # 1. Route
    query_type = route_query(query)

    # 2. Retrieve
    chunks = retrieve(query, query_type=query_type, top_k=top_k)

    # 3. Generate
    answer = generate(query, chunks, stream=stream)

    result = {
        "query": query,
        "query_type": query_type,
        "answer": answer or "Error: Could not generate a response.",
        "sources": [],
    }

    if show_sources:
        result["sources"] = [
            {
                "title": c["title"],
                "entity_type": c["entity_type"],
                "chunk_index": c["chunk_index"],
                "distance": round(c["distance"], 4),
                "text": c["text"][:200] + "…" if len(c["text"]) > 200 else c["text"],
            }
            for c in chunks
        ]

    return result


# ════════════════════════════════════════════════════════════
#  5.  BAĞIMSIZ TEST (CLI)
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  🧪  RAG PIPELINE – İnteraktif Test")
    print("=" * 60)
    print("  Çıkmak için 'quit' veya 'exit' yazın.\n")

    while True:
        query = input("  ❓  Sorunuz: ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            print("  👋  Çıkılıyor…")
            break

        query_type = route_query(query)
        print(f"  🔀  Router kararı: {query_type}")

        print(f"  🔍  Retrieve ediliyor (top-{TOP_K})…")
        chunks = retrieve(query, query_type=query_type)
        for i, c in enumerate(chunks, 1):
            print(f"      [{i}] {c['title']} ({c['entity_type']})  "
                  f"dist={c['distance']:.4f}")

        print(f"\n  🤖  Cevap üretiliyor (stream)…\n")
        print("  ─" * 30)
        answer = generate(query, chunks, stream=True)
        print("  ─" * 30)
        print()
