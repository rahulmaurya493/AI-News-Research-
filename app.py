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

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "NewsRAG AI",
    page_icon  = "🗞",
    layout     = "wide"
)

# ─────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url(\'https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300;1,400&display=swap\');

:root {
    --bg:      #080c10;
    --bg2:     #0d1117;
    --bg3:     #111820;
    --border:  rgba(255,255,255,0.07);
    --border2: rgba(255,255,255,0.13);
    --accent:  #e8c97a;
    --accent2: #c4a95e;
    --teal:    #4dd9c0;
    --white:   #f0ece4;
    --muted:   rgba(240,236,228,0.40);
    --muted2:  rgba(240,236,228,0.18);
}

html, body, .stApp {
    font-family: \'DM Mono\', monospace;
    background: var(--bg) !important;
    color: var(--white) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; }

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border2) !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

.sidebar-header {
    padding: 28px 22px 18px 22px;
    border-bottom: 1px solid var(--border2);
}
.sidebar-logo {
    font-family: \'Syne\', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--white);
    letter-spacing: -0.02em;
}
.sidebar-logo span { color: var(--accent); }
.sidebar-sub {
    font-size: 0.64rem;
    color: var(--muted);
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-top: 5px;
}
.sidebar-section {
    padding: 16px 22px 4px 22px;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
}

/* INPUTS */
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--white) !important;
    font-family: \'DM Mono\', monospace !important;
    font-size: 0.82rem !important;
    caret-color: var(--accent) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232,201,122,0.12) !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--white) !important;
    font-family: \'DM Mono\', monospace !important;
    font-size: 0.82rem !important;
}
label {
    color: var(--muted) !important;
    font-size: 0.70rem !important;
    font-family: \'DM Mono\', monospace !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* BUTTONS */
.stButton > button {
    font-family: \'Syne\', sans-serif !important;
    font-size: 0.80rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    border: 1px solid var(--border2) !important;
    background: rgba(255,255,255,0.04) !important;
    color: var(--white) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(232,201,122,0.10) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
.primary-btn .stButton > button {
    background: var(--accent) !important;
    color: #080c10 !important;
    border: none !important;
    font-weight: 800 !important;
}
.primary-btn .stButton > button:hover {
    background: var(--accent2) !important;
    color: #080c10 !important;
    box-shadow: 0 4px 18px rgba(232,201,122,0.28) !important;
}

/* NAV RADIO */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] > div { gap: 0 !important; flex-direction: column !important; }
div[data-testid="stRadio"] > div > label {
    font-family: \'DM Mono\', monospace !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    padding: 11px 22px !important;
    border-radius: 0 !important;
    border-left: 2px solid transparent !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
div[data-testid="stRadio"] > div > label:hover {
    color: var(--white) !important;
    background: rgba(255,255,255,0.04) !important;
    border-left-color: var(--accent) !important;
}

/* MAIN LAYOUT */
.main-pad { padding: 36px 44px; }

.page-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border2);
}
.page-title {
    font-family: \'Syne\', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--white);
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.page-title span { color: var(--accent); }
.page-sub {
    font-family: \'Crimson Pro\', serif;
    font-size: 1rem;
    font-style: italic;
    color: var(--muted);
    margin-top: 5px;
}
.live-badge {
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: \'DM Mono\', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--teal);
    background: rgba(77,217,192,0.07);
    border: 1px solid rgba(77,217,192,0.20);
    padding: 6px 14px;
    border-radius: 4px;
}
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--teal);
    animation: blink 2s infinite;
}
@keyframes blink {
    0%,100% { opacity:1; } 50% { opacity:0.3; }
}

/* METRIC STRIP */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 12px;
    margin-bottom: 28px;
}
.metric-tile {
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
}
.metric-tile::before {
    content: \'\';
    position: absolute; top:0; left:0; right:0; height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.metric-num {
    font-family: \'Syne\', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--white);
    line-height: 1;
}
.metric-key {
    font-size: 0.6rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 7px;
}

/* SECTION HEADING */
.sec-head {
    font-family: \'DM Mono\', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 28px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-head::after { content:\'\'; flex:1; height:1px; background:var(--border2); }

/* Q&A */
.q-block {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    padding: 18px 0 14px 0;
    border-bottom: 1px solid var(--border);
}
.q-chip {
    font-family: \'DM Mono\', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    color: var(--accent);
    background: rgba(232,201,122,0.10);
    border: 1px solid rgba(232,201,122,0.25);
    border-radius: 4px;
    padding: 4px 8px;
    white-space: nowrap;
    margin-top: 2px;
    flex-shrink: 0;
}
.q-text {
    font-family: \'Syne\', sans-serif;
    font-size: 1.0rem;
    font-weight: 700;
    color: var(--white);
    line-height: 1.4;
}
.a-block {
    background: linear-gradient(135deg, rgba(232,201,122,0.05), rgba(77,217,192,0.02));
    border: 1px solid rgba(232,201,122,0.14);
    border-radius: 10px;
    padding: 22px 26px;
    margin: 10px 0 14px 0;
    position: relative;
    overflow: hidden;
}
.a-block::before {
    content: \'"\';
    position: absolute; top:-12px; left:16px;
    font-family: \'Crimson Pro\', serif;
    font-size: 6rem;
    color: rgba(232,201,122,0.06);
    line-height: 1;
    pointer-events: none;
}
.a-label {
    font-size: 0.58rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 10px;
    font-family: \'DM Mono\', monospace;
}
.a-text {
    font-family: \'Crimson Pro\', serif;
    font-size: 1.08rem;
    line-height: 1.85;
    color: var(--white);
}

/* SOURCE CARDS */
.src-card {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.src-num {
    font-family: \'DM Mono\', monospace;
    font-size: 0.63rem;
    color: var(--accent);
    background: rgba(232,201,122,0.10);
    border-radius: 4px;
    padding: 2px 7px;
    flex-shrink: 0;
    margin-top: 2px;
}
.src-title {
    font-family: \'Syne\', sans-serif;
    font-size: 0.84rem;
    font-weight: 600;
    color: var(--white);
    margin-bottom: 4px;
    line-height: 1.3;
}
.src-meta {
    font-family: \'DM Mono\', monospace;
    font-size: 0.67rem;
    color: var(--muted);
}
.src-score { color: var(--teal); }

/* TAGS */
.entity-tag {
    display: inline-block;
    background: rgba(232,201,122,0.07);
    border: 1px solid rgba(232,201,122,0.22);
    color: var(--accent);
    padding: 4px 12px;
    border-radius: 3px;
    font-size: 0.72rem;
    margin: 3px 3px 3px 0;
    font-family: \'DM Mono\', monospace;
}
.topic-tag {
    display: inline-block;
    background: rgba(77,217,192,0.07);
    border: 1px solid rgba(77,217,192,0.22);
    color: var(--teal);
    padding: 4px 12px;
    border-radius: 3px;
    font-size: 0.72rem;
    margin: 3px 3px 3px 0;
    font-family: \'DM Mono\', monospace;
}

/* STEPS */
.step-grid {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 12px;
    margin: 20px 0;
}
.step-card {
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 22px 18px;
    position: relative;
    overflow: hidden;
}
.step-card::after {
    content: attr(data-n);
    position: absolute; bottom:-10px; right:10px;
    font-family: \'Syne\', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    color: rgba(232,201,122,0.05);
    line-height: 1;
    pointer-events: none;
}
.step-icon { font-size: 1.5rem; margin-bottom: 10px; }
.step-title {
    font-family: \'Syne\', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--white);
    margin-bottom: 5px;
}
.step-desc {
    font-size: 0.72rem;
    color: var(--muted);
    line-height: 1.55;
    font-family: \'DM Mono\', monospace;
}

/* EMPTY STATE */
.empty-state {
    text-align: center;
    padding: 60px 40px;
    border: 1px dashed var(--border2);
    border-radius: 12px;
    background: var(--bg3);
}
.empty-icon { font-size: 2.8rem; margin-bottom: 14px; }
.empty-title {
    font-family: \'Syne\', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--white);
    margin-bottom: 8px;
}
.empty-sub {
    font-size: 0.78rem;
    color: var(--muted);
    font-family: \'DM Mono\', monospace;
    line-height: 1.65;
}

/* ALERTS */
.stAlert {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    font-family: \'DM Mono\', monospace !important;
    font-size: 0.78rem !important;
}
div[data-testid="stAlert"][kind="success"] {
    border-color: rgba(77,217,192,0.28) !important;
    color: var(--teal) !important;
}

/* STATUS */
div[data-testid="stStatus"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 10px !important;
}

/* EXPANDER */
details {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    margin-bottom: 8px !important;
}
summary {
    font-family: \'DM Mono\', monospace !important;
    font-size: 0.78rem !important;
    color: var(--white) !important;
    padding: 12px 16px !important;
}

/* SCROLLBAR */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.10); border-radius: 3px; }

/* SPINNER */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* FOOTER */
.site-footer {
    margin-top: 60px;
    padding-top: 18px;
    border-top: 1px solid var(--border);
    font-family: \'DM Mono\', monospace;
    font-size: 0.63rem;
    color: var(--muted2);
    letter-spacing: 0.08em;
    display: flex;
    justify-content: space-between;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
for key, val in {
    "pipeline_ready" : False,
    "articles"       : [],
    "documents"      : [],
    "faiss_index"    : None,
    "embed_model"    : None,
    "groq_client"    : None,
    "nlp_model"      : None,
    "chat_history"   : [],
    "current_query"  : ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────
@st.cache_resource
def load_embed_model():
    return get_embedding_model()

@st.cache_resource
def load_nlp():
    return load_nlp_model()


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-logo">🗞 News<span>RAG</span></div>
            <div class="sidebar-sub">Retrieval-Augmented News Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(\'<div class="sidebar-section">API Keys</div>\', unsafe_allow_html=True)
        news_key = st.text_input("NewsAPI Key", type="password",
                                  placeholder="Enter NewsAPI key…")
        groq_key = st.text_input("Groq API Key", type="password",
                                  placeholder="Enter Groq key…")

        st.markdown(\'<div class="sidebar-section">News Query</div>\', unsafe_allow_html=True)
        query = st.text_input("Topic", placeholder="e.g.  RBI interest rates India",
                               value=st.session_state.current_query)

        col1, col2 = st.columns(2)
        with col1:
            max_articles = st.selectbox("Articles", [10, 15, 20], index=0)
        with col2:
            days_back = st.selectbox("Days back", [3, 7, 14], index=1)

        st.markdown("<div style=\'height:10px\'></div>", unsafe_allow_html=True)
        st.markdown(\'<div class="primary-btn">\', unsafe_allow_html=True)
        build_btn = st.button("⚡  Fetch & Build Pipeline", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.pipeline_ready:
            st.success(
                f"Pipeline ready · {len(st.session_state.articles)} articles · "
                f"{len(st.session_state.documents)} chunks"
            )

        st.markdown(\'<div class="sidebar-section" style="margin-top:14px">Navigate</div>\',
                    unsafe_allow_html=True)
        page = st.radio("nav", ["🤖  Ask Questions", "🔍  NLP Analysis", "🗂️  History"],
                        label_visibility="collapsed")

        st.markdown(\'<div class="sidebar-section" style="margin-top:14px">Quick Questions</div>\',
                    unsafe_allow_html=True)
        for q in ["What are the latest developments?", "Who are the key people involved?",
                  "What impact does this have?", "What happened recently?"]:
            if st.button(q, use_container_width=True, key=f"sample_{q}"):
                st.session_state["sample_q"] = q

    return news_key, groq_key, query, max_articles, days_back, build_btn, page


# ─────────────────────────────────────────
# BUILD PIPELINE
# ─────────────────────────────────────────
def build_pipeline(news_key, groq_key, query, max_articles, days_back):
    final_news_key = news_key or st.secrets.get("news_key", "")
    final_groq_key = groq_key or st.secrets.get("groq_key", "")

    if not final_news_key or not final_groq_key:
        st.error("Enter API keys in sidebar or Streamlit Secrets.")
        return
    if not query:
        st.error("Enter a news topic to fetch.")
        return

    with st.status("Building RAG pipeline…", expanded=True) as status:
        st.write("Fetching news articles…")
        news_client = get_news_client(news_key)
        articles    = fetch_articles(news_client, query=query,
                                     days_back=days_back, max_articles=max_articles)
        if not articles:
            st.error("No articles found. Try a different topic or range.")
            return
        st.write(f"✓  {len(articles)} articles fetched")

        st.write("Chunking articles…")
        documents = chunk_all_articles(articles)
        st.write(f"✓  {len(documents)} chunks created")

        st.write("Embedding chunks…")
        embed_model      = load_embed_model()
        _, vectors, docs = embed_documents(embed_model, documents)
        st.write(f"✓  {len(docs)} chunks embedded")

        st.write("Building FAISS index…")
        faiss_index = build_faiss_index(vectors)
        st.write(f"✓  FAISS ready ({faiss_index.ntotal} vectors)")

        st.write("Connecting Groq LLM…")
        groq_client = get_groq_client(final_groq_key)
        st.write("✓  LLM connected")

        st.session_state.update({
            "articles": articles, "documents": docs,
            "faiss_index": faiss_index, "embed_model": embed_model,
            "groq_client": groq_client, "pipeline_ready": True,
            "current_query": query, "chat_history": []
        })
        status.update(label="Pipeline ready — ask your question.", state="complete")


# ─────────────────────────────────────────
# PAGE 1 — ASK
# ─────────────────────────────────────────
def page_ask():
    st.markdown(\'<div class="main-pad">\', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
        <div>
            <div class="page-title">News<span>RAG</span></div>
            <div class="page-sub">Ask anything — AI answers from live news</div>
        </div>
        <div class="live-badge"><div class="live-dot"></div>Live Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_ready:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">👈</div>
            <div class="empty-title">Build the Pipeline First</div>
            <div class="empty-sub">Enter your API keys and news topic in the sidebar,<br>
            then click Fetch & Build Pipeline.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(\'<div class="sec-head">How It Works</div>\', unsafe_allow_html=True)
        st.markdown("""
        <div class="step-grid">
            <div class="step-card" data-n="1"><div class="step-icon">📰</div>
                <div class="step-title">Fetch</div>
                <div class="step-desc">Live articles pulled from NewsAPI</div></div>
            <div class="step-card" data-n="2"><div class="step-icon">✂️</div>
                <div class="step-title">Chunk</div>
                <div class="step-desc">Articles split into searchable segments</div></div>
            <div class="step-card" data-n="3"><div class="step-icon">🔢</div>
                <div class="step-title">Embed</div>
                <div class="step-desc">Segments vectorised for semantic search</div></div>
            <div class="step-card" data-n="4"><div class="step-icon">🤖</div>
                <div class="step-title">Answer</div>
                <div class="step-desc">Groq LLM answers from relevant context</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Metrics
    st.markdown(f"""
    <div class="metric-strip">
        <div class="metric-tile"><div class="metric-num">{len(st.session_state.articles)}</div><div class="metric-key">Articles</div></div>
        <div class="metric-tile"><div class="metric-num">{len(st.session_state.documents)}</div><div class="metric-key">Chunks</div></div>
        <div class="metric-tile"><div class="metric-num">{len(st.session_state.chat_history)}</div><div class="metric-key">Questions</div></div>
        <div class="metric-tile"><div class="metric-num">RAG</div><div class="metric-key">Mode</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(\'<div class="sec-head">Ask a Question</div>\', unsafe_allow_html=True)

    # Sample Q pre-fill
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    if st.session_state.get("sample_q"):
        st.session_state["question_input"] = st.session_state.pop("sample_q")

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        question = st.text_input("Q", placeholder="e.g.  What did the RBI announce about interest rates?",
                                  label_visibility="collapsed", key="question_input")
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
        st.markdown(\'<div class="sec-head">Answers</div>\', unsafe_allow_html=True)

    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"""
        <div class="q-block">
            <div class="q-chip">QUERY</div>
            <div class="q-text">{chat["question"]}</div>
        </div>
        <div class="a-block">
            <div class="a-label">AI · Sourced from live news</div>
            <div class="a-text">{chat["answer"]}</div>
        </div>
        """, unsafe_allow_html=True)

        if chat["sources"]:
            st.markdown(\'<div class="sec-head" style="margin-top:4px">Sources</div>\',
                        unsafe_allow_html=True)
            for i, src in enumerate(chat["sources"][:3], 1):
                st.markdown(f"""
                <div class="src-card">
                    <div class="src-num">#{i}</div>
                    <div>
                        <div class="src-title">{src["title"][:88]}…</div>
                        <div class="src-meta">{src["source"]} &nbsp;·&nbsp; {src["date"]}
                        &nbsp;·&nbsp; <span class="src-score">Match {src["score"]}%</span></div>
                    </div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<div style=\'height:20px\'></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE 2 — NLP
# ─────────────────────────────────────────
def page_nlp():
    st.markdown(\'<div class="main-pad">\', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
        <div>
            <div class="page-title">NLP <span>Analysis</span></div>
            <div class="page-sub">Entities and trends extracted from news</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_ready:
        st.markdown("""<div class="empty-state">
            <div class="empty-icon">🔍</div>
            <div class="empty-title">No Data Yet</div>
            <div class="empty-sub">Build the pipeline first from the sidebar.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    nlp_model = load_nlp()
    with st.spinner("Extracting entities…"):
        all_entities = extract_from_articles(nlp_model, st.session_state.articles)
        topics       = get_top_topics(st.session_state.articles, top_n=15)
        formatted    = format_entities_display(all_entities)

    st.markdown(\'<div class="sec-head">Named Entities</div>\', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (label, items) in enumerate(formatted.items()):
        with cols[i % 2]:
            st.markdown(f"<div style=\'font-family:DM Mono,monospace;font-size:0.60rem;letter-spacing:0.16em;text-transform:uppercase;color:rgba(240,236,228,0.35);margin-bottom:8px\'>{label}</div>",
                        unsafe_allow_html=True)
            tags = " ".join([f"<span class=\'entity-tag\'>{item}</span>" for item in items])
            st.markdown(f"<div style=\'margin-bottom:20px\'>{tags}</div>", unsafe_allow_html=True)

    st.markdown(\'<div class="sec-head">Trending Topics</div>\', unsafe_allow_html=True)
    topic_tags = " ".join([
        f"<span class=\'topic-tag\'>{w} <span style=\'opacity:0.45\'>×{c}</span></span>"
        for w, c in topics
    ])
    st.markdown(f"<div style=\'margin-bottom:24px\'>{topic_tags}</div>", unsafe_allow_html=True)

    st.markdown(\'<div class="sec-head">All Articles</div>\', unsafe_allow_html=True)
    for art in st.session_state.articles:
        st.markdown(f"""
        <div class="src-card">
            <div>
                <div class="src-title">{art["title"][:88]}</div>
                <div class="src-meta">{art["source"]} &nbsp;·&nbsp; {art["published_at"]}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE 3 — HISTORY
# ─────────────────────────────────────────
def page_history():
    st.markdown(\'<div class="main-pad">\', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
        <div>
            <div class="page-title">Chat <span>History</span></div>
            <div class="page-sub">All questions and answers this session</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""<div class="empty-state">
            <div class="empty-icon">🗂️</div>
            <div class="empty-title">No History Yet</div>
            <div class="empty-sub">Ask questions on the Ask page to build your history.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    col_a, col_b = st.columns([4, 1])
    with col_a:
        st.markdown(
            f"<span style=\'font-family:DM Mono,monospace;font-size:0.75rem;color:rgba(240,236,228,0.35)\'>"
            f"{len(st.session_state.chat_history)} questions this session</span>",
            unsafe_allow_html=True)
    with col_b:
        if st.button("Clear All", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("<div style=\'height:14px\'></div>", unsafe_allow_html=True)
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        n = len(st.session_state.chat_history) - i
        with st.expander(f"Q{n}  ·  {chat[\'question\'][:65]}…"):
            st.markdown(f"**Question:** {chat[\'question\']}")
            st.markdown(f"**Answer:** {chat[\'answer\']}")
            if chat["sources"]:
                st.markdown("**Sources:**")
                for src in chat["sources"]:
                    st.markdown(f"— {src[\'source\']} · {src[\'date\']}")

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    news_key, groq_key, query, max_articles, days_back, build_btn, page = render_sidebar()

    if build_btn:
        build_pipeline(news_key, groq_key, query, max_articles, days_back)

    if "Ask" in page:
        page_ask()
    elif "NLP" in page:
        page_nlp()
    elif "History" in page:
        page_history()

    st.markdown("""
    <div class="main-pad" style="padding-top:0">
        <div class="site-footer">
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
