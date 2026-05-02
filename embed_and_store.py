# ============================================================
# embed_and_store.py  –  Ollama Embedding + ChromaDB Storage
# ============================================================
#  • MySQL'den chunk'ları okur
#  • Ollama nomic-embed-text ile embedding üretir (yerel, API yok)
#  • Tek bir ChromaDB koleksiyonuna metadata ile yazar (Option B)
# ============================================================

import os
import sys
import time
import sqlite3
import requests
import chromadb
from config import (
    SQLITE_DB_PATH,
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
)

# ── ChromaDB depolama dizini ─────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "wiki_rag"

# ── Embedding batch boyutu (Ollama'ya bir seferde gönderilecek) ──
BATCH_SIZE = 32


# ════════════════════════════════════════════════════════════
#  1.  OLLAMA EMBEDDING  (native HTTP – framework yok)
# ════════════════════════════════════════════════════════════

def ollama_embed(texts: list[str], model: str = OLLAMA_EMBED_MODEL) -> list[list[float]]:
    """
    Ollama /api/embed endpoint'ine istek atar ve embedding vektörlerini döner.

    Ollama, tek bir istek içinde birden fazla metin kabul eder (batch).
    Dönen JSON:  { "embeddings": [[0.1, 0.2, ...], ...] }
    """
    url = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {
        "model": model,
        "input": texts,
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])

        if len(embeddings) != len(texts):
            print(f"  [WARN] Beklenen {len(texts)} embedding, dönen {len(embeddings)}")

        return embeddings

    except requests.RequestException as exc:
        print(f"  [ERROR] Ollama embedding isteği başarısız: {exc}")
        sys.exit(1)


# ════════════════════════════════════════════════════════════
#  2.  SQLITE'DAN CHUNK'LARI OKU
# ════════════════════════════════════════════════════════════

def fetch_chunks_from_db() -> list[dict]:
    """
    SQLite chunks tablosundan tüm chunk kayıtlarını döner.
    Her kayıt: {id, document_id, chunk_index, chunk_text, entity_type, title}
    """
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"  [!] SQLite veritabanı bulunamadı: {SQLITE_DB_PATH}")
        print("      Önce 'python ingest.py' çalıştırın.")
        sys.exit(1)

    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row   # dict-benzeri erişim
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, document_id, chunk_index, chunk_text,
               char_count, entity_type, title
        FROM chunks
        ORDER BY document_id, chunk_index
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    cursor.close()
    conn.close()
    return rows


# ════════════════════════════════════════════════════════════
#  3.  CHROMADB'YE KAYDET  (Tek koleksiyon + metadata)
# ════════════════════════════════════════════════════════════

def get_chroma_collection() -> chromadb.Collection:
    """
    Kalıcı (persistent) ChromaDB client oluşturur ve
    'wiki_rag' koleksiyonunu döner.  Koleksiyon yoksa yaratılır.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Varsa sil, temiz basla  (idempotent re-run)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"  [DEL] Eski koleksiyon silindi: {COLLECTION_NAME}")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # Cosine benzerlik
    )
    print(f"  [OK] Koleksiyon olusturuldu: {COLLECTION_NAME}  (cosine)\n")
    return collection


def store_in_chroma(chunks: list[dict], collection: chromadb.Collection):
    """
    Chunk'lari batch'ler halinde embed edip ChromaDB'ye yazar.

    Her chunk icin metadata:
      - entity_type : "person" | "place"
      - title       : Orijinal Wikipedia basligi
      - chunk_index : Dokuman icindeki sira numarasi
      - document_id : MySQL document ID
    """
    total = len(chunks)
    stored = 0

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["chunk_text"] for c in batch]

        print(f"  [..] Embedding batch {i // BATCH_SIZE + 1}  "
              f"({i+1}-{min(i+len(batch), total)} / {total}) ...")

        t0 = time.time()
        embeddings = ollama_embed(texts)
        elapsed = time.time() - t0
        print(f"      Embedding suresi: {elapsed:.1f}s")

        # -- ChromaDB'ye ekle --
        ids = [f"chunk_{c['id']}" for c in batch]
        metadatas = [
            {
                "entity_type": c["entity_type"],
                "title": c["title"],
                "chunk_index": c["chunk_index"],
                "document_id": c["document_id"],
            }
            for c in batch
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        stored += len(batch)
        print(f"      ChromaDB'ye eklendi: {stored}/{total}\n")

    return stored


# ================================================================
#  4.  DOGRULAMA
# ================================================================

def verify_collection(collection: chromadb.Collection):
    """ChromaDB koleksiyonundaki verileri dogrular."""
    count = collection.count()
    print("=" * 60)
    print("  DOGRULAMA")
    print("=" * 60)
    print(f"  Toplam vektor sayisi : {count}")

    # Tip dagilimi
    all_data = collection.get(include=["metadatas"])
    types = {}
    titles = set()
    for m in all_data["metadatas"]:
        t = m["entity_type"]
        types[t] = types.get(t, 0) + 1
        titles.add(m["title"])

    for t, c in sorted(types.items()):
        print(f"  {t:>8s}  chunks : {c}")
    print(f"  Benzersiz baslik     : {len(titles)}")

    # Ornek sorgu — Chroma'nin default embedder'i 384-dim MiniLM,
    # bizimkiler 768-dim Ollama. Bu yuzden sorguyu da Ollama ile embed et.
    print("\n  Ornek sorgu: 'Who is Albert Einstein?'")
    query_embedding = ollama_embed(["Who is Albert Einstein?"])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    for j, (doc, meta, dist) in enumerate(
        zip(results["documents"][0],
            results["metadatas"][0],
            results["distances"][0])
    ):
        print(f"\n  -- Sonuc {j+1}  (distance: {dist:.4f}) --")
        print(f"  Baslik : {meta['title']}  ({meta['entity_type']})")
        print(f"  Metin  : {doc[:120]}...")

    print("\n" + "=" * 60)


# ================================================================
#  5.  ANA AKIS
# ================================================================

def run_embed_and_store():
    print("=" * 60)
    print("  EMBED & STORE PIPELINE")
    print("=" * 60 + "\n")

    # 1. SQLite'dan chunk'lari oku
    print("  [..] SQLite'dan chunk'lar okunuyor...")
    chunks = fetch_chunks_from_db()
    if not chunks:
        print("  [!] Hic chunk bulunamadi! Once 'python ingest.py' calistirin.")
        sys.exit(1)
    print(f"  [OK] {len(chunks)} chunk okundu.\n")

    # 2. ChromaDB koleksiyonunu hazirla
    collection = get_chroma_collection()

    # 3. Embed & store
    stored = store_in_chroma(chunks, collection)

    # 4. Dogrulama
    verify_collection(collection)

    print(f"\n  [OK] Toplam {stored} chunk basariyla ChromaDB'ye kaydedildi.")
    print(f"  [DIR] Veri dizini: {CHROMA_PERSIST_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_embed_and_store()
