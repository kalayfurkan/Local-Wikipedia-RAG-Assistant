# ============================================================
# ingest.py  –  Wikipedia Ingest & Native Chunking Pipeline
# ============================================================
#  • Wikipedia API'den düz metin çeker
#  • Native Python ile chunk'lar (overlap destekli)
#  • MySQL'e yazar  (documents + chunks tabloları)
# ============================================================

import re
import time
import sqlite3
import requests
from config import (
    SQLITE_DB_PATH,
    WIKIPEDIA_API_URL, CHUNK_SIZE, CHUNK_OVERLAP,
    PEOPLE, PLACES,
)


# ════════════════════════════════════════════════════════════
#  1.  WIKIPEDIA FETCHER
# ════════════════════════════════════════════════════════════

def fetch_wikipedia_text(title: str, max_retries: int = 5) -> str | None:
    """
    Wikipedia REST API uzerinden sayfanin duz metnini ceker.
    429 (rate-limit) hatalarinda exponential backoff ile yeniden dener.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json",
        "redirects": 1,
    }
    headers = {
        "User-Agent": "WikiRAG/1.0 (Local RAG Project; educational use)"
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(WIKIPEDIA_API_URL, params=params,
                                headers=headers, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32 saniye
                print(f"      [429] Rate limited, {wait}s bekleniyor... "
                      f"(deneme {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    print(f"  [WARN] Sayfa bulunamadi: {title}")
                    return None
                return page_data.get("extract", "")
        except requests.RequestException as exc:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"      [RETRY] Hata: {exc}, {wait}s bekleniyor...")
                time.sleep(wait)
            else:
                print(f"  [ERROR] Wikipedia istegi basarisiz - {title}: {exc}")
                return None
    print(f"  [ERROR] Max retry asildi - {title}")
    return None


# ════════════════════════════════════════════════════════════
#  2.  METİN TEMİZLEME
# ════════════════════════════════════════════════════════════

def clean_text(raw: str) -> str:
    """
    Wikipedia metnini temizler:
      – Çoklu boşluk / tab → tek boşluk
      – Fazla boş satırları azalt
      – Baş-son boşlukları kırp
    """
    text = raw.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)            # Çoklu boşluk → tek
    text = re.sub(r"\n{3,}", "\n\n", text)         # 3+ boş satır → 2
    return text.strip()


# ════════════════════════════════════════════════════════════
#  3.  NATIVE PYTHON CHUNKING  (Overlap destekli)
# ════════════════════════════════════════════════════════════

def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Metni sabit boyutlu parçalara böler (overlap ile).

    Algoritma
    ---------
    1.  Metni cümlelere böl  ('. ', '? ', '! ' sınırlarında).
    2.  Cümleleri birleştirerek chunk oluştur.
        – Bir chunk chunk_size'ı aştığında yeni chunk başlat.
        – Önceki chunk'ın son 'overlap' karakterlik kısmını yeni chunk'a
          başlangıç olarak ekle (context sürekliliği).
    3.  Her chunk için {index, text, char_count} dön.

    Neden cümle tabanlı?
    --------------------
    Karakter ortasından kesmek anlam kaybına yol açar.
    Cümle sınırlarına saygı duyan bir strateji, retrieval
    kalitesini artırır.
    """
    if not text:
        return []

    # ── Cümlelere ayır ──────────────────────────────────────
    # Basit ama etkili regex: Nokta/soru/ünlem + boşluktan sonra
    # büyük harfle başlayan yeni cümle.  Kısaltmalar (Dr. Mr.)
    # gibi edge-case'leri mükemmel yakalamaz ama yeterli.
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)

    chunks: list[dict] = []
    current_chunk = ""
    chunk_index = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Mevcut chunk'a cümleyi eklemeyi dene
        candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Mevcut chunk'ı kaydet (boş değilse)
            if current_chunk:
                chunks.append({
                    "index": chunk_index,
                    "text": current_chunk,
                    "char_count": len(current_chunk),
                })
                chunk_index += 1

                # ── Overlap: önceki chunk'ın son 'overlap' karakteri ──
                if overlap > 0 and len(current_chunk) > overlap:
                    # Overlap bölgesini kelime sınırında başlat
                    tail = current_chunk[-overlap:]
                    # Kelime ortasında kesmemek için ilk boşluğa ilerle
                    space_pos = tail.find(" ")
                    if space_pos != -1:
                        tail = tail[space_pos + 1:]
                    current_chunk = tail + " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Tek cümle chunk_size'dan büyük → zorla kaydet
                current_chunk = sentence

    # Son chunk'ı kaydet
    if current_chunk.strip():
        chunks.append({
            "index": chunk_index,
            "text": current_chunk.strip(),
            "char_count": len(current_chunk.strip()),
        })

    return chunks


# ════════════════════════════════════════════════════════════
#  4.  SQLITE SETUP & STORAGE
# ════════════════════════════════════════════════════════════

def get_db_connection():
    """SQLite bağlantısı oluşturur ve foreign key desteğini açar."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def setup_database():
    """
    SQLite dosyasını ve tablolarını oluşturur.
    SQLite tek dosya olduğu için ayrıca CREATE DATABASE adımı yok.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # ── documents tablosu ────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT      NOT NULL UNIQUE,
            entity_type TEXT      NOT NULL CHECK(entity_type IN ('person','place')),
            full_text   TEXT      NOT NULL,
            char_count  INTEGER   NOT NULL,
            chunk_count INTEGER   DEFAULT 0,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── chunks tablosu ───────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id  INTEGER NOT NULL,
            chunk_index  INTEGER NOT NULL,
            chunk_text   TEXT    NOT NULL,
            char_count   INTEGER NOT NULL,
            entity_type  TEXT    NOT NULL CHECK(entity_type IN ('person','place')),
            title        TEXT    NOT NULL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            UNIQUE (document_id, chunk_index)
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print(f"[OK] SQLite veritabanı hazır: {SQLITE_DB_PATH}\n")


def save_document(title: str, entity_type: str, full_text: str,
                  chunks: list[dict]):
    """
    Dokümanı ve chunk'larını SQLite'a yazar.
    Aynı başlık varsa günceller (UPSERT – ON CONFLICT).
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # ── Dokümanı yaz / güncelle ──────────────────────────────
    cursor.execute("""
        INSERT INTO documents (title, entity_type, full_text, char_count, chunk_count)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(title) DO UPDATE SET
            full_text   = excluded.full_text,
            char_count  = excluded.char_count,
            chunk_count = excluded.chunk_count
    """, (title, entity_type, full_text, len(full_text), len(chunks)))

    # Doküman ID'sini al
    cursor.execute("SELECT id FROM documents WHERE title = ?", (title,))
    doc_id = cursor.fetchone()[0]

    # ── Eski chunk'ları sil, yenilerini yaz ──────────────────
    cursor.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))

    if chunks:
        insert_sql = """
            INSERT INTO chunks
                (document_id, chunk_index, chunk_text, char_count, entity_type, title)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        rows = [
            (doc_id, c["index"], c["text"], c["char_count"], entity_type, title)
            for c in chunks
        ]
        cursor.executemany(insert_sql, rows)

    conn.commit()
    cursor.close()
    conn.close()


# ════════════════════════════════════════════════════════════
#  5.  ANA PIPELINE
# ════════════════════════════════════════════════════════════

def is_already_ingested(title: str) -> bool:
    """
    SQLite'da bu başlık için zaten chunk'lı bir doküman var mı?
    Re-run idempotent olsun diye: ingest.py tekrar çalıştırıldığında
    Wikipedia'ya gereksiz istek atmayalım, sadece eksikleri çekelim.
    """
    conn = get_db_connection()
    row = conn.execute(
        "SELECT chunk_count FROM documents WHERE title = ?", (title,)
    ).fetchone()
    conn.close()
    return bool(row and row[0] > 0)


def ingest_entity(title: str, entity_type: str) -> str:
    """
    Tek bir Wikipedia varlığını ingest eder. Zaten DB'de ise dokunmaz.
    Dönüş: 'skipped' | 'fetched' | 'failed'  (run_ingest sleep yönetimi için)
    """
    if is_already_ingested(title):
        print(f"  [SKIP] Zaten DB'de: {title}\n")
        return "skipped"

    print(f"  [>] Fetching: {title} ...")
    raw = fetch_wikipedia_text(title)
    if not raw:
        print(f"  [!] Atlandi: {title}\n")
        return "failed"

    text = clean_text(raw)
    print(f"      Temiz metin: {len(text):,} karakter")

    chunks = chunk_text(text)
    print(f"      Chunk sayisi: {len(chunks)}  "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    save_document(title, entity_type, text, chunks)
    print(f"  [OK] SQLite'a kaydedildi: {title}\n")
    return "fetched"


def run_ingest():
    """Tum kisi ve yerleri ingest eder. Re-run idempotent: zaten DB'de
    olanları atlar, sadece eksikleri Wikipedia'dan çeker."""
    setup_database()

    print("=" * 60)
    print("  FAMOUS PEOPLE  (20)")
    print("=" * 60)
    for name in PEOPLE:
        status = ingest_entity(name, "person")
        if status == "fetched":
            time.sleep(2)   # Wikipedia rate-limit'e saygi (sadece gerçek fetch sonrası)

    print("=" * 60)
    print("  FAMOUS PLACES  (20)")
    print("=" * 60)
    for name in PLACES:
        status = ingest_entity(name, "place")
        if status == "fetched":
            time.sleep(2)

    # -- Ozet istatistik --
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(char_count) FROM documents")
    total_chars = cursor.fetchone()[0] or 0

    cursor.close()
    conn.close()

    print("\n" + "=" * 60)
    print("  INGEST TAMAMLANDI")
    print("=" * 60)
    print(f"  Dokuman sayisi  : {doc_count}")
    print(f"  Toplam chunk    : {chunk_count}")
    print(f"  Toplam karakter : {total_chars:,}")
    print("=" * 60)


# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_ingest()
