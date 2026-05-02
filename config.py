# ============================================================
# config.py  –  Central Configuration for Local RAG System
# ============================================================

import os

# ── SQLite ───────────────────────────────────────────────────
# Tek dosyalık yerel veritabanı; ekstra daemon/parola gerektirmez.
SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), "wiki_rag.db")

# ── Ollama ───────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3.2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE = 500          # Her chunk'taki karakter sayısı
CHUNK_OVERLAP = 50        # Chunk'lar arası örtüşme (karakter)

# ── Wikipedia ────────────────────────────────────────────────
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# ── Entities ─────────────────────────────────────────────────
PEOPLE = [
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    "Mahatma Gandhi",
    "Cleopatra",
    "Isaac Newton",
    "Wolfgang Amadeus Mozart",
    "Napoleon Bonaparte",
    "Martin Luther King Jr.",
    "Charles Darwin",
    "Pablo Picasso",
    "Aristotle",
    "Elon Musk",
]

PLACES = [
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Pyramids of Giza",
    "Mount Everest",
    "Petra",
    "Stonehenge",
    "Angkor Wat",
    "Niagara Falls",
    "Santorini",
    "Great Barrier Reef",
    "Galata Tower",
    "Christ the Redeemer",
    "Chichen Itza",
    "Mount Fuji",
]
