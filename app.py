code = '''
import streamlit as st
import sys, os
sys.path.append("src")

from news_fetcher   import get_news_client, fetch_articles, fetch_top_headlines
from chunker        import chunk_all_articles
from embeddings     import get_embedding_model, embed_documents, embed_query
from vector_store   import build_faiss_index, search_index, save_faiss_index
from rag_pipeline   import get_groq_client, run_rag_pipeline, setup_pipeline
from nlp_extractor  import load_nlp_model, extract_from_articles, get_top_topics, format_entities_display

st.set_page_config(
    page_title="NewsRAG AI",
    page_icon="🗞",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url(\'https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Crimson+Pro:ital,wght@0,400;0,600;1,400&display=swap\');

/* ── TOKENS ── */
:root {
    --bg:       #07090d;
    --bg2:      #0c1018;
    --bg3:      #111620;
    --bg4:      #161d2a;
    --border:   rgba(255,255,255,0.06);
    --border2:  rgba(255,255,255,0.11);
    --gold:     #d4a843;
    --gold-lt:  #e8c97a;
    --gold-dim: rgba(212,168,67,0.12);
    --teal:     #3ecfb8;
    --teal-dim: rgba(62,207,184,0.08);
    --white:    #eceae3;
    --muted:    rgba(236,234,227,0.38);
    --muted2:   rgba(236,234,227,0.14);
    --fs-xs:    0.68rem;
    --fs-sm:    0.78rem;
    --fs-md:    0.88rem;
    --fs-lg:    1.0rem;
    --fs-xl:    1.2rem;
    --fs-2xl:   1.6rem;
    --fs-3xl:   2.0rem;
    --radius:   10px;
    --radius-sm: 6px;
}

/* ── RESET ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: \'DM Mono\', monospace !important;
    background: var(--bg) !important;
    color: var(--white) !important;
    font-size: 14px;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── SIDEBAR — ALWAYS VISIBLE ── */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border2) !important;
    min-width: 270px !important;
    max-width: 270px !important;
    transform: none !important;
    visibility: visible !important;
    display: block !important;
    overflow-y: auto;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
    overflow: visible !important;
}
/* Hide collapse arrow */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* ── SIDEBAR CONTENT ── */
.sb-header {
    padding: 22px 20px 16px 20px;
    border-bottom: 1px solid var(--border2);
    background: linear-gradient(180deg, rgba(212,168,67,0.04) 0%, transparent 100%);
}
.sb-logo {
    font-family: \'Syne\', sans-serif;
    font-size: var(--fs-xl);
    font-weight: 800;
    color: var(--white);
    letter-spacing: -0.02em;
    line-height: 1;
}
.sb-logo em { color: var(--gold-lt); font-style: normal; }
.sb-tagline {
    font-size: 0.60rem;
    color: var(--muted);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 6px;
    line-height: 1;
}
.sb-section {
    padding: 14px 20px 6px 20px;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
    font-family: \'DM Mono\', monospace;
}
.sb-divider {
    height: 1px;
    background: var(--border2);
    margin: 8px 0;
}

/* ── FORM INPUTS ── */
.stTextInput > div > div > input {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--white) !important;
    font-family: \'DM Mono\', monospace !important;
    font-size: var(--fs-sm) !important;
    padding: 8px 12px !important;
    height: 36px !important;
    caret-color: var(--gold-lt) !important;
    transition: border-color 0.15s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(212,168,67,0.10) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--muted2) !important;
}

.stSelectbox > div > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--white) !important;
    font-family: \'DM Mono\', monospace !important;
    font-size: var(--fs-sm) !important;
    min-height: 36px !important;
}
.stSelectbox > div > div > div {
    padding: 4px 8px !important;
}

/* Input labels */
.stTextInput label, .stSelectbox label, .stTextArea label {
    font-family: \'DM Mono\', monospace !important;
    font-size: 0.62rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 5px !important;
}

/* Reduce gap between label and input */
.stTextInput, .stSelectbox { margin-bottom: 10px !important; }

/* ── ALL BUTTONS BASE ── */
.stButton > button {
    font-family: \'DM Mono\', monospace !important;
    font-size: var(--fs-sm) !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border2) !important;
    background: var(--bg3) !important;
    color: var(--muted) !important;
    padding: 7px 14px !important;
    height: 36px !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
    text-align: left !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    border-color: rgba(212,168,67,0.40) !important;
    color: var(--white) !important;
    background: var(--bg4) !important;
}

/* ── PRIMARY (Fetch & Build) ── */
.btn-primary .stButton > button {
    background: var(--gold) !important;
    color: #07090d !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: var(--fs-sm) !important;
    letter-spacing: 0.06em !important;
    height: 40px !important;
    border-radius: var(--radius-sm) !important;
    text-align: center !important;
}
.btn-primary .stButton > button:hover {
    background: #e8c97a !important;
    color: #07090d !important;
    box-shadow: 0 2px 12px rgba(212,168,67,0.30) !important;
    transform: translateY(-1px) !important;
}

/* ── NAV BUTTONS ── */
.nav-item .stButton > button {
    background: transparent !important;
    border: none !important;
    border-left: 2px solid transparent !important;
    border-radius: 0 !important;
    color: var(--muted) !important;
    font-size: var(--fs-sm) !important;
    font-weight: 400 !important;
    padding: 10px 20px !important;
    height: auto !important;
    letter-spacing: 0.01em !important;
    text-align: left !important;
}
.nav-item .stButton > button:hover {
    background: rgba(255,255,255,0.03) !important;
    border-left-color: rgba(212,168,67,0.35) !important;
    color: var(--white) !important;
    border-radius: 0 !important;
}
.nav-item-active .stButton > button {
    background: rgba(212,168,67,0.06) !important;
    border: none !important;
    border-left: 2px solid var(--gold) !important;
    border-radius: 0 !important;
    color: var(--gold-lt) !important;
    font-size: var(--fs-sm) !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    height: auto !important;
    letter-spacing: 0.01em !important;
    text-align: left !important;
    box-shadow: none !important;
}
.nav-item-active .stButton > button:hover {
    background: rgba(212,168,67,0.09) !important;
    color: var(--gold-lt) !important;
    border-left-color: var(--gold) !important;
}

/* ── SAMPLE Q BUTTONS ── */
.btn-sample .stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--muted) !important;
    font-size: 0.70rem !important;
    font-weight: 400 !important;
    padding: 6px 12px !important;
    height: auto !important;
    margin-bottom: 4px !important;
    text-align: left !important;
}
.btn-sample .stButton > button:hover {
    border-color: rgba(212,168,67,0.30) !important;
    color: var(--white) !important;
    background: var(--bg3) !important;
}

/* ── MAIN CONTENT AREA ── */
.main-wrap {
    padding: 32px 40px 60px 40px;
}

/* ── PAGE HEADER ── */
.pg-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding-bottom: 18px;
    margin-bottom: 24px;
    border-bottom: 1px solid var(--border2);
}
.pg-title {
    font-family: \'Syne\', sans-serif;
    font-size: var(--fs-3xl);
    font-weight: 800;
    color: var(--white);
    letter-spacing: -0.03em;
    line-height: 1;
}
.pg-title em { color: var(--gold-lt); font-style: normal; }
.pg-sub {
    font-family: \'Crimson Pro\', serif;
    font-size: 0.95rem;
    font-style: italic;
    color: var(--muted);
    margin-top: 6px;
    line-height: 1;
}
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: \'DM Mono\', monospace;
    font-size: 0.60rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--teal);
    background: var(--teal-dim);
    border: 1px solid rgba(62,207,184,0.18);
    padding: 5px 12px;
    border-radius: 3px;
}
.live-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--teal);
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

/* ── SECTION LABEL ── */
.sec-label {
    font-family: \'DM Mono\', monospace;
    font-size: 0.60rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted2);
    margin: 24px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label::after { content:\'\'; flex:1; height:1px; background:var(--border2); }

/* ── METRIC STRIP ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 10px;
    margin-bottom: 24px;
}
.metric-cell {
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
}
.metric-cell::before {
    content:\'\';
    position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, var(--gold), transparent);
}
.metric-val {
    font-family: \'Syne\', sans-serif;
    font-size: var(--fs-2xl);
    font-weight: 800;
    color: var(--white);
    line-height: 1;
}
.metric-lbl {
    font-family: \'DM Mono\', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 6px;
}

/* ── Q&A BLOCKS ── */
.q-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 16px 0 12px 0;
    border-bottom: 1px solid var(--border);
}
.q-badge {
    font-family: \'DM Mono\', monospace;
    font-size: 0.56rem;
    letter-spacing: 0.12em;
    color: var(--gold);
    background: var(--gold-dim);
    border: 1px solid rgba(212,168,67,0.22);
    border-radius: 3px;
    padding: 3px 7px;
    white-space: nowrap;
    flex-shrink: 0;
    margin-top: 3px;
}
.q-text {
    font-family: \'Syne\', sans-serif;
    font-size: var(--fs-lg);
    font-weight: 700;
    color: var(--white);
    line-height: 1.4;
}
.a-box {
    background: linear-gradient(135deg, rgba(212,168,67,0.04) 0%, rgba(62,207,184,0.02) 100%);
    border: 1px solid rgba(212,168,67,0.12);
    border-radius: var(--radius);
    padding: 20px 22px;
    margin: 10px 0 12px 0;
    position: relative;
    overflow: hidden;
}
.a-box::before {
    content: \'"\';
    position: absolute; top:-14px; left:14px;
    font-family: \'Crimson Pro\', serif;
    font-size: 5rem;
    color: rgba(212,168,67,0.05);
    line-height: 1;
    pointer-events: none;
}
.a-tag {
    font-family: \'DM Mono\', monospace;
    font-size: 0.56rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 8px;
}
.a-text {
    font-family: \'Crimson Pro\', serif;
    font-size: 1.05rem;
    line-height: 1.80;
    color: var(--white);
}

/* ── SOURCE CARDS ── */
.src-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 11px 14px;
    margin-bottom: 7px;
}
.src-idx {
    font-family: \'DM Mono\', monospace;
    font-size: 0.60rem;
    color: var(--gold);
    background: var(--gold-dim);
    border-radius: 3px;
    padding: 2px 6px;
    flex-shrink: 0;
    margin-top: 2px;
}
.src-ttl {
    font-family: \'Syne\', sans-serif;
    font-size: var(--fs-sm);
    font-weight: 600;
    color: var(--white);
    margin-bottom: 3px;
    line-height: 1.35;
}
.src-meta {
    font-family: \'DM Mono\', monospace;
    font-size: 0.65rem;
    color: var(--muted);
}
.src-score { color: var(--teal); }

/* ── TAGS ── */
.tag-ent {
    display: inline-block;
    background: var(--gold-dim);
    border: 1px solid rgba(212,168,67,0.20);
    color: var(--gold-lt);
    padding: 3px 10px;
    border-radius: 3px;
    font-family: \'DM Mono\', monospace;
    font-size: 0.70rem;
    margin: 2px 2px 2px 0;
}
.tag-topic {
    display: inline-block;
    background: var(--teal-dim);
    border: 1px solid rgba(62,207,184,0.20);
    color: var(--teal);
    padding: 3px 10px;
    border-radius: 3px;
    font-family: \'DM Mono\', monospace;
    font-size: 0.70rem;
    margin: 2px 2px 2px 0;
}

/* ── STEP CARDS ── */
.step-row {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 10px;
    margin: 18px 0;
}
.step-cell {
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 20px 16px;
    position: relative;
    overflow: hidden;
}
.step-cell::after {
    content: attr(data-n);
    position: absolute; bottom:-10px; right:8px;
    font-family: \'Syne\', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: rgba(212,168,67,0.04);
    line-height: 1;
    pointer-events: none;
}
.step-ico { font-size: 1.4rem; margin-bottom: 10px; }
.step-ttl {
    font-family: \'Syne\', sans-serif;
    font-size: var(--fs-md);
    font-weight: 700;
    color: var(--white);
    margin-bottom: 4px;
}
.step-dsc {
    font-family: \'DM Mono\', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    line-height: 1.55;
}

/* ── EMPTY STATE ── */
.empty-wrap {
    text-align: center;
    padding: 56px 40px;
    border: 1px dashed var(--border2);
    border-radius: var(--radius);
    background: var(--bg3);
    margin: 8px 0;
}
.empty-ico  { font-size: 2.4rem; margin-bottom: 12px; }
.empty-ttl  {
    font-family: \'Syne\', sans-serif;
    font-size: var(--fs-xl);
    font-weight: 800;
    color: var(--white);
    margin-bottom: 8px;
}
.empty-sub  {
    font-family: \'DM Mono\', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    line-height: 1.65;
}

/* ── STREAMLIT ALERTS ── */
.stAlert {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important;
    font-family: \'DM Mono\', monospace !important;
    font-size: var(--fs-sm) !important;
}
div[data-testid="stAlert"][kind="success"] {
    border-color: rgba(62,207,184,0.25) !important;
    color: var(--teal) !important;
}

/* ── STATUS BOX ── */
div[data-testid="stStatus"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    font-family: \'DM Mono\', monospace !important;
    font-size: var(--fs-sm) !important;
}

/* ── EXPANDER ── */
details {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    margin-bottom: 7px !important;
}
summary {
    font-family: \'DM Mono\', monospace !important;
    font-size: var(--fs-sm) !important;
    color: var(--white) !important;
    padding: 11px 14px !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: var(--gold) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 2px; }

/* ── FOOTER ── */
.pg-footer {
    padding-top: 20px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: \'DM Mono\', monospace;
    font-size: 0.62rem;
    color: var(--muted2);
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE ───────────────────────
for k, v in {
    "pipeline_ready": False, "articles": [], "documents": [],
    "faiss_index": None, "embed_model": None, "groq_client": None,
    "chat_history": [], "current_query": "", "page": "ask",
    "question_input": ""
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── CACHED LOADERS ─────────────────────
@st.cache_resource
def load_embed_model():
    return get_embedding_model()

@st.cache_resource
def load_nlp():
    return load_nlp_model()


# ─── SIDEBAR ────────────────────────────
def render_sidebar():
    with st.sidebar:

        # Logo
        st.markdown("""
        <div class="sb-header">
            <div class="sb-logo">News<em>RAG</em></div>
            <div class="sb-tagline">Retrieval-Augmented News AI</div>
        </div>
        """, unsafe_allow_html=True)

        # API Keys
        st.markdown(\'<div class="sb-section">API Keys</div>\', unsafe_allow_html=True)
        news_key = st.text_input("NewsAPI Key", type="password", placeholder="sk-…")
        groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk-…")

        # Query
        st.markdown(\'<div class="sb-section">News Topic</div>\', unsafe_allow_html=True)
        query = st.text_input("Search Query", placeholder="e.g. RBI interest rates India",
                               value=st.session_state.current_query)
        col1, col2 = st.columns(2)
        with col1:
            max_articles = st.selectbox("Articles", [10, 15, 20], index=0)
        with col2:
            days_back = st.selectbox("Days back", [3, 7, 14], index=1)

        st.markdown("<div style=\'height:8px\'></div>", unsafe_allow_html=True)
        st.markdown(\'<div class="btn-primary">\', unsafe_allow_html=True)
        build_btn = st.button("⚡  Fetch & Build Pipeline", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.pipeline_ready:
            st.success(
                f"✓  Ready · {len(st.session_state.articles)} articles "
                f"· {len(st.session_state.documents)} chunks"
            )

        # Navigation
        st.markdown(\'<div class="sb-divider"></div>\', unsafe_allow_html=True)
        st.markdown(\'<div class="sb-section">Pages</div>\', unsafe_allow_html=True)

        nav_pages = [
            ("ask",     "🤖  Ask Questions"),
            ("nlp",     "🔍  NLP Analysis"),
            ("history", "🗂️  Chat History"),
        ]
        for key, label in nav_pages:
            active = st.session_state["page"] == key
            cls = "nav-item-active" if active else "nav-item"
            st.markdown(f\'<div class="{cls}">\', unsafe_allow_html=True)
            if st.button(label, use_container_width=True, key=f"nav_{key}"):
                st.session_state["page"] = key
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # Quick Questions
        st.markdown(\'<div class="sb-divider"></div>\', unsafe_allow_html=True)
        st.markdown(\'<div class="sb-section">Quick Questions</div>\', unsafe_allow_html=True)
        for q in [
            "What are the latest developments?",
            "Who are the key people involved?",
            "What is the impact?",
            "What happened recently?"
        ]:
            st.markdown(\'<div class="btn-sample">\', unsafe_allow_html=True)
            if st.button(q, use_container_width=True, key=f"sq_{q}"):
                st.session_state["question_input"] = q
                st.session_state["page"] = "ask"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    return news_key, groq_key, query, max_articles, days_back, build_btn


# ─── BUILD PIPELINE ─────────────────────
def build_pipeline(news_key, groq_key, query, max_articles, days_back):
    nk = news_key or st.secrets.get("news_key", "")
    gk = groq_key or st.secrets.get("groq_key", "")
    if not nk or not gk:
        st.error("Enter both API keys in the sidebar.")
        return
    if not query:
        st.error("Enter a news topic to search.")
        return

    with st.status("Building RAG pipeline…", expanded=True) as status:
        st.write("Fetching articles…")
        client   = get_news_client(nk)
        articles = fetch_articles(client, query=query, days_back=days_back, max_articles=max_articles)
        if not articles:
            st.error("No articles found. Try a different topic or wider date range.")
            return
        st.write(f"✓  {len(articles)} articles fetched")

        st.write("Chunking…")
        documents = chunk_all_articles(articles)
        st.write(f"✓  {len(documents)} chunks created")

        st.write("Embedding…")
        embed_model      = load_embed_model()
        _, vectors, docs = embed_documents(embed_model, documents)
        st.write(f"✓  {len(docs)} vectors generated")

        st.write("Building FAISS index…")
        faiss_index = build_faiss_index(vectors)
        st.write(f"✓  Index ready ({faiss_index.ntotal} vectors)")

        st.write("Connecting Groq LLM…")
        groq_client = get_groq_client(gk)
        st.write("✓  LLM connected")

        st.session_state.update({
            "articles": articles, "documents": docs,
            "faiss_index": faiss_index, "embed_model": embed_model,
            "groq_client": groq_client, "pipeline_ready": True,
            "current_query": query, "chat_history": []
        })
        status.update(label="Pipeline ready — ask your question below.", state="complete")


# ─── PAGE: ASK ──────────────────────────
def page_ask():
    st.markdown(\'<div class="main-wrap">\', unsafe_allow_html=True)
    st.markdown("""
    <div class="pg-header">
        <div>
            <div class="pg-title">News<em>RAG</em></div>
            <div class="pg-sub">Ask anything — AI answers from live news</div>
        </div>
        <div class="live-pill"><div class="live-dot"></div>Live Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_ready:
        st.markdown("""
        <div class="empty-wrap">
            <div class="empty-ico">👈</div>
            <div class="empty-ttl">Build the Pipeline First</div>
            <div class="empty-sub">Enter your API keys and a news topic in the sidebar,<br>
            then click "Fetch & Build Pipeline" to get started.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(\'<div class="sec-label">How It Works</div>\', unsafe_allow_html=True)
        st.markdown("""
        <div class="step-row">
            <div class="step-cell" data-n="1">
                <div class="step-ico">📰</div>
                <div class="step-ttl">Fetch</div>
                <div class="step-dsc">Live articles pulled from NewsAPI</div>
            </div>
            <div class="step-cell" data-n="2">
                <div class="step-ico">✂️</div>
                <div class="step-ttl">Chunk</div>
                <div class="step-dsc">Articles split into searchable segments</div>
            </div>
            <div class="step-cell" data-n="3">
                <div class="step-ico">🔢</div>
                <div class="step-ttl">Embed</div>
                <div class="step-dsc">Segments vectorised for semantic search</div>
            </div>
            <div class="step-cell" data-n="4">
                <div class="step-ico">🤖</div>
                <div class="step-ttl">Answer</div>
                <div class="step-dsc">Groq LLM answers from top results</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Stats
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-cell"><div class="metric-val">{len(st.session_state.articles)}</div><div class="metric-lbl">Articles</div></div>
        <div class="metric-cell"><div class="metric-val">{len(st.session_state.documents)}</div><div class="metric-lbl">Chunks</div></div>
        <div class="metric-cell"><div class="metric-val">{len(st.session_state.chat_history)}</div><div class="metric-lbl">Questions</div></div>
        <div class="metric-cell"><div class="metric-val">RAG</div><div class="metric-lbl">Pipeline Mode</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(\'<div class="sec-label">Ask a Question</div>\', unsafe_allow_html=True)

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        question = st.text_input(
            "Question", label_visibility="collapsed",
            placeholder="e.g.  What did the RBI announce about interest rates?",
            key="question_input"
        )
    with col_btn:
        ask_btn = st.button("Ask ⚡", use_container_width=True)

    if ask_btn and question:
        with st.spinner("Searching and generating answer…"):
            result = run_rag_pipeline(
                question=question, groq_client=st.session_state.groq_client,
                embedding_model=st.session_state.embed_model,
                faiss_index=st.session_state.faiss_index,
                documents=st.session_state.documents, top_k=5
            )
            st.session_state.chat_history.append({
                "question": question, "answer": result["answer"],
                "sources": result["sources"], "chunks": result["chunks"]
            })

    if st.session_state.chat_history:
        st.markdown(\'<div class="sec-label">Answers</div>\', unsafe_allow_html=True)

    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"""
        <div class="q-row">
            <div class="q-badge">QUERY</div>
            <div class="q-text">{chat["question"]}</div>
        </div>
        <div class="a-box">
            <div class="a-tag">AI Answer · Sourced from live news</div>
            <div class="a-text">{chat["answer"]}</div>
        </div>
        """, unsafe_allow_html=True)

        if chat["sources"]:
            st.markdown(\'<div class="sec-label" style="margin-top:6px">Sources</div>\', unsafe_allow_html=True)
            for i, src in enumerate(chat["sources"][:3], 1):
                st.markdown(f"""
                <div class="src-row">
                    <div class="src-idx">#{i}</div>
                    <div>
                        <div class="src-ttl">{src["title"][:88]}…</div>
                        <div class="src-meta">
                            {src["source"]} &nbsp;·&nbsp; {src["date"]}
                            &nbsp;·&nbsp; <span class="src-score">Match {src["score"]}%</span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<div style=\'height:18px\'></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─── PAGE: NLP ──────────────────────────
def page_nlp():
    st.markdown(\'<div class="main-wrap">\', unsafe_allow_html=True)
    st.markdown("""
    <div class="pg-header">
        <div>
            <div class="pg-title">NLP <em>Analysis</em></div>
            <div class="pg-sub">Entities and trends extracted from news</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_ready:
        st.markdown("""<div class="empty-wrap">
            <div class="empty-ico">🔍</div>
            <div class="empty-ttl">No Data Yet</div>
            <div class="empty-sub">Build the pipeline first using the sidebar.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    nlp_model = load_nlp()
    with st.spinner("Extracting entities…"):
        all_entities = extract_from_articles(nlp_model, st.session_state.articles)
        topics       = get_top_topics(st.session_state.articles, top_n=15)
        formatted    = format_entities_display(all_entities)

    st.markdown(\'<div class="sec-label">Named Entities</div>\', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (label, items) in enumerate(formatted.items()):
        with cols[i % 2]:
            st.markdown(f"<div style=\'font-family:DM Mono,monospace;font-size:0.58rem;letter-spacing:0.16em;text-transform:uppercase;color:var(--muted2);margin-bottom:8px\'>{label}</div>",
                        unsafe_allow_html=True)
            tags = " ".join([f"<span class=\'tag-ent\'>{item}</span>" for item in items])
            st.markdown(f"<div style=\'margin-bottom:20px\'>{tags}</div>", unsafe_allow_html=True)

    st.markdown(\'<div class="sec-label">Trending Topics</div>\', unsafe_allow_html=True)
    topic_tags = " ".join([
        f"<span class=\'tag-topic\'>{w} <span style=\'opacity:0.4\'>×{c}</span></span>"
        for w, c in topics
    ])
    st.markdown(f"<div style=\'margin-bottom:22px\'>{topic_tags}</div>", unsafe_allow_html=True)

    st.markdown(\'<div class="sec-label">All Articles</div>\', unsafe_allow_html=True)
    for art in st.session_state.articles:
        st.markdown(f"""
        <div class="src-row">
            <div>
                <div class="src-ttl">{art["title"][:90]}</div>
                <div class="src-meta">{art["source"]} &nbsp;·&nbsp; {art["published_at"]}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─── PAGE: HISTORY ──────────────────────
def page_history():
    st.markdown(\'<div class="main-wrap">\', unsafe_allow_html=True)
    st.markdown("""
    <div class="pg-header">
        <div>
            <div class="pg-title">Chat <em>History</em></div>
            <div class="pg-sub">All questions and answers this session</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""<div class="empty-wrap">
            <div class="empty-ico">🗂️</div>
            <div class="empty-ttl">No History Yet</div>
            <div class="empty-sub">Ask questions on the Ask page to build up history.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    col_a, col_b = st.columns([5, 1])
    with col_a:
        st.markdown(
            f"<span style=\'font-family:DM Mono,monospace;font-size:0.68rem;"
            f"color:var(--muted)\'>{len(st.session_state.chat_history)} questions this session</span>",
            unsafe_allow_html=True)
    with col_b:
        if st.button("Clear All", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("<div style=\'height:12px\'></div>", unsafe_allow_html=True)
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        n = len(st.session_state.chat_history) - i
        with st.expander(f"Q{n}  ·  {chat[\'question\'][:60]}…"):
            st.markdown(f"**Question:** {chat[\'question\']}")
            st.markdown(f"**Answer:** {chat[\'answer\']}")
            if chat["sources"]:
                st.markdown("**Sources:**")
                for src in chat["sources"]:
                    st.markdown(f"— {src[\'source\']} · {src[\'date\']}")

    st.markdown("</div>", unsafe_allow_html=True)


# ─── MAIN ───────────────────────────────
def main():
    news_key, groq_key, query, max_articles, days_back, build_btn = render_sidebar()

    if build_btn:
        build_pipeline(news_key, groq_key, query, max_articles, days_back)

    page = st.session_state.get("page", "ask")
    if page == "ask":
        page_ask()
    elif page == "nlp":
        page_nlp()
    elif page == "history":
        page_history()

    st.markdown("""
    <div style=\'padding: 0 40px\'>
        <div class="pg-footer">
            <span>NewsRAG AI · Rahul Maurya</span>
            <span>LangChain · FAISS · Groq · SpaCy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
'''

with open("app.py", "w") as f:
    f.write(code)
print("✅ app.py created!")
