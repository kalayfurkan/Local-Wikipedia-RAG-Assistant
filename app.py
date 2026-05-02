# ============================================================
# app.py  –  Streamlit Chat UI for Local Wikipedia RAG
# ============================================================
#  • ChatGPT benzeri sohbet arayüzü
#  • Oturum belleği (chat history)
#  • Kaynak chunk gösterimi (opsiyonel)
#  • Sistem durumu sidebar'ı
# ============================================================

import time
import streamlit as st
import requests
from rag_pipeline import ask, route_query, retrieve, TOP_K
from config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL


# ════════════════════════════════════════════════════════════
#  SAYFA AYARLARI
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="WikiRAG – Local AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Genel ────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Ana başlık ───────────────────────────────── */
    .main-title {
        text-align: center;
        padding: 1rem 0 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* ── Sidebar ──────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #667eea !important;
    }

    /* ── Kaynak kartları ──────────────────────────── */
    .source-card {
        background: #f8f9ff;
        border-left: 3px solid #667eea;
        padding: 0.6rem 0.8rem;
        margin: 0.4rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        line-height: 1.4;
    }
    .source-card .src-title {
        font-weight: 600;
        color: #667eea;
    }
    .source-card .src-type {
        display: inline-block;
        background: #667eea22;
        color: #667eea;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 500;
        margin-left: 4px;
    }
    .source-card .src-dist {
        color: #999;
        font-size: 0.72rem;
    }

    /* ── Route badge ──────────────────────────────── */
    .route-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .route-person { background: #e8f5e9; color: #2e7d32; }
    .route-place  { background: #e3f2fd; color: #1565c0; }
    .route-both   { background: #fff3e0; color: #e65100; }

    /* ── Durum göstergesi ─────────────────────────── */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-ok   { background: #4caf50; }
    .status-fail { background: #f44336; }

    /* ── Footer ───────────────────────────────────── */
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.75rem;
        padding: 2rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True


# ════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ════════════════════════════════════════════════════════════

def check_ollama_status() -> bool:
    """Ollama sunucusunun çalışıp çalışmadığını kontrol eder."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def route_badge_html(query_type: str) -> str:
    """Router kararı için renkli badge HTML'i döner."""
    cls = f"route-{query_type}"
    labels = {"person": "👤 Person", "place": "🏛️ Place", "both": "🔄 Both"}
    return f'<span class="route-badge {cls}">{labels.get(query_type, query_type)}</span>'


def source_card_html(src: dict) -> str:
    """Kaynak chunk için kart HTML'i döner."""
    return (
        f'<div class="source-card">'
        f'<span class="src-title">{src["title"]}</span>'
        f'<span class="src-type">{src["entity_type"]}</span> '
        f'<span class="src-dist">distance: {src["distance"]}</span>'
        f'<br><span style="color:#555">{src["text"]}</span>'
        f'</div>'
    )


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🧠 WikiRAG")
    st.markdown("Local AI Wikipedia Assistant")
    st.markdown("---")

    # Sistem durumu
    st.markdown("### ⚙️ System Status")
    ollama_ok = check_ollama_status()
    dot = "status-ok" if ollama_ok else "status-fail"
    status_text = "Online" if ollama_ok else "Offline"
    st.markdown(
        f'<span class="status-dot {dot}"></span> Ollama: **{status_text}**',
        unsafe_allow_html=True,
    )
    st.markdown(f"**LLM:** `{OLLAMA_LLM_MODEL}`")
    st.markdown(f"**Embeddings:** `{OLLAMA_EMBED_MODEL}`")

    st.markdown("---")

    # Ayarlar
    st.markdown("### 🎛️ Settings")
    st.session_state.show_sources = st.toggle(
        "Show source chunks", value=st.session_state.show_sources
    )
    top_k = st.slider("Retrieved chunks (Top-K)", 1, 10, TOP_K)

    st.markdown("---")

    # Sohbeti temizle
    if st.button("🗑️  Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Örnek sorular
    st.markdown("### 💡 Example Questions")
    examples = [
        "Who was Albert Einstein?",
        "Where is the Eiffel Tower?",
        "Compare Messi and Ronaldo",
        "Which famous place is in Turkey?",
        "Who is the president of Mars?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.pending_question = ex
            st.rerun()

    st.markdown(
        '<div class="footer">Built with Ollama · ChromaDB · Streamlit<br>'
        'No external APIs used</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
#  ANA İÇERİK
# ════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-title">🧠 WikiRAG</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">'
    'Local Wikipedia RAG Assistant — powered by Ollama · ChromaDB · Streamlit'
    '</p>',
    unsafe_allow_html=True,
)

# ── Geçmiş mesajları göster ──────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Route badge
            if "query_type" in msg:
                st.markdown(route_badge_html(msg["query_type"]),
                            unsafe_allow_html=True)
            st.markdown(msg["content"])
            # Kaynak kartları
            if st.session_state.show_sources and msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        st.markdown(source_card_html(src),
                                    unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])


# ── Pending (sidebar'dan gelen) soru kontrolü ────────────────
if "pending_question" in st.session_state:
    prompt = st.session_state.pop("pending_question")
else:
    prompt = st.chat_input("Ask me anything about famous people & places…")


# ── Yeni soru işle ───────────────────────────────────────────
if prompt:
    # Kullanıcı mesajını göster ve kaydet
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistan yanıtı
    with st.chat_message("assistant"):
        # Ollama kontrolü
        if not check_ollama_status():
            err = "⚠️ Ollama is not running. Please start Ollama first."
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err}
            )
        else:
            # Route
            query_type = route_query(prompt)
            st.markdown(route_badge_html(query_type), unsafe_allow_html=True)

            with st.spinner("🔍 Retrieving & generating…"):
                t0 = time.time()
                result = ask(
                    query=prompt,
                    top_k=top_k,
                    stream=False,
                    show_sources=True,
                )
                elapsed = time.time() - t0

            # Cevap
            st.markdown(result["answer"])
            st.caption(f"⏱️ {elapsed:.1f}s")

            # Kaynaklar
            if st.session_state.show_sources and result.get("sources"):
                with st.expander(f"📚 Sources ({len(result['sources'])})"):
                    for src in result["sources"]:
                        st.markdown(source_card_html(src),
                                    unsafe_allow_html=True)

            # Session'a kaydet
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "query_type": result["query_type"],
                "sources": result.get("sources", []),
            })
