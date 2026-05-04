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
    page_icon  = "🗞️",
    layout     = "wide"
)

# ─────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap\');
html, body, .stApp { font-family: \'Inter\', sans-serif; background: #0d0d1a; color: #e0e0f0; }

section[data-testid="stSidebar"] {
    background: #13131f !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 14px;
}

.answer-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.30);
    border-radius: 14px;
    padding: 22px 26px;
    margin: 14px 0;
    line-height: 1.7;
}

.source-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid #6366f1;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

.entity-tag {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    color: #a5b4fc;
    padding: 3px 12px;
    border-radius: 50px;
    font-size: 0.80rem;
    margin: 3px;
    font-weight: 600;
}

.topic-tag {
    display: inline-block;
    background: rgba(0,220,130,0.12);
    border: 1px solid rgba(0,220,130,0.30);
    color: #6ee7b7;
    padding: 3px 12px;
    border-radius: 50px;
    font-size: 0.80rem;
    margin: 3px;
    font-weight: 600;
}

.metric-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-val { font-size: 1.7rem; font-weight: 800; color: #fff; }
.metric-lbl { font-size: 0.70rem; color: rgba(255,255,255,0.40);
              text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px; }

h1,h2,h3 { color: #ffffff !important; }
label { color: rgba(255,255,255,0.75) !important; font-weight: 600 !important; }

.stButton > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    padding: 0.5rem 1.5rem !important;
}
.stTextInput input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
}

.footer { color:rgba(255,255,255,0.2); font-size:0.72rem;
          text-align:center; margin-top:40px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE
# WHY: Streamlit reruns entire script on every
#      interaction — session_state persists data
#      between reruns so we dont re-fetch news!
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
# CACHED MODEL LOADERS
# WHY: @st.cache_resource loads model ONCE
#      and reuses it — without this, model
#      reloads on every interaction = very slow!
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
        st.markdown("## 🗞️ NewsRAG AI")
        st.markdown("*Ask questions about real news*")
        st.markdown("---")

        # API Keys
        st.markdown("### 🔑 API Keys")
        news_key = st.text_input(
            "NewsAPI Key",
            type     = "password",
            placeholder = "Enter NewsAPI key..."
        )
        groq_key = st.text_input(
            "Groq API Key",
            type     = "password",
            placeholder = "Enter Groq key..."
        )

        st.markdown("---")

        # Topic
        st.markdown("### 🔍 News Topic")
        query = st.text_input(
            "What news to fetch?",
            placeholder = "e.g. RBI interest rates India",
            value       = st.session_state.current_query
        )

        col1, col2 = st.columns(2)
        with col1:
            max_articles = st.selectbox(
                "Articles", [10, 15, 20], index=0
            )
        with col2:
            days_back = st.selectbox(
                "Days back", [3, 7, 14], index=1
            )

        # Build Pipeline Button
        st.markdown("<br>", unsafe_allow_html=True)
        build_btn = st.button(
            "🚀 Fetch & Build Pipeline",
            use_container_width=True
        )

        # Status
        if st.session_state.pipeline_ready:
            st.success(f"✅ Pipeline ready!")
            st.caption(
                f"📰 {len(st.session_state.articles)} articles | "
                f"✂️ {len(st.session_state.documents)} chunks"
            )

        st.markdown("---")

        # Sample questions
        st.markdown("### 💡 Sample Questions")
        samples = [
            "What are the latest developments?",
            "Who are the key people involved?",
            "What impact does this have?",
            "What happened recently?"
        ]
        for q in samples:
            if st.button(q, use_container_width=True, key=f"sample_{q}"):
                st.session_state["sample_q"] = q

    return news_key, groq_key, query, max_articles, days_back, build_btn


# ─────────────────────────────────────────
# BUILD PIPELINE
# ─────────────────────────────────────────
def build_pipeline(news_key, groq_key, query, max_articles, days_back):
    """Fetch news + build full RAG pipeline."""

    if not news_key or not groq_key:
        st.error("❌ Please enter both API keys in sidebar!")
        return

    if not query:
        st.error("❌ Please enter a news topic!")
        return

    with st.status("🔧 Building RAG pipeline...", expanded=True) as status:

        # Step 1
        st.write("📰 Fetching news articles...")
        news_client = get_news_client(news_key)
        articles    = fetch_articles(
            news_client, query=query,
            days_back=days_back, max_articles=max_articles
        )

        if not articles:
            st.error("❌ No articles found! Try different topic or days.")
            return

        st.write(f"✅ Got {len(articles)} articles!")

        # Step 2
        st.write("✂️  Chunking articles...")
        documents = chunk_all_articles(articles)
        st.write(f"✅ Created {len(documents)} chunks!")

        # Step 3
        st.write("🔢 Embedding chunks...")
        embed_model          = load_embed_model()
        _, vectors, docs     = embed_documents(embed_model, documents)
        st.write(f"✅ Embedded {len(docs)} chunks!")

        # Step 4
        st.write("🗄️  Building FAISS index...")
        faiss_index = build_faiss_index(vectors)
        st.write(f"✅ FAISS index ready ({faiss_index.ntotal} vectors)!")

        # Step 5
        st.write("🤖 Connecting Groq LLM...")
        groq_client = get_groq_client(groq_key)
        st.write("✅ LLM connected!")

        # Save to session state
        st.session_state.articles       = articles
        st.session_state.documents      = docs
        st.session_state.faiss_index    = faiss_index
        st.session_state.embed_model    = embed_model
        st.session_state.groq_client    = groq_client
        st.session_state.pipeline_ready = True
        st.session_state.current_query  = query
        st.session_state.chat_history   = []

        status.update(
            label="✅ Pipeline ready! Ask your question below.",
            state="complete"
        )


# ─────────────────────────────────────────
# PAGE 1 — ASK QUESTIONS (main page)
# ─────────────────────────────────────────
def page_ask():
    st.title("🗞️ NewsRAG AI")
    st.markdown("*Ask any question — AI answers from real news articles*")

    if not st.session_state.pipeline_ready:
        st.markdown("""
        <div class=\'card\' style=\'text-align:center;padding:40px;\'>
            <div style=\'font-size:3rem;\'>👈</div>
            <h3>Get Started</h3>
            <p style=\'color:rgba(255,255,255,0.5);\'>
            Enter your API keys and news topic in the sidebar,
            then click "Fetch & Build Pipeline"
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Show how it works
        st.markdown("### 🔄 How It Works")
        c1,c2,c3,c4 = st.columns(4)
        steps = [
            ("📰","Fetch News","Real articles fetched from NewsAPI"),
            ("✂️","Chunk","Articles split into searchable pieces"),
            ("🔢","Embed","Pieces converted to vectors"),
            ("🤖","Answer","LLM answers from relevant pieces"),
        ]
        for col, (icon, title, desc) in zip([c1,c2,c3,c4], steps):
            with col:
                st.markdown(f"""
                <div class=\'metric-box\'>
                    <div style=\'font-size:2rem;\'>{icon}</div>
                    <div style=\'font-weight:800;margin:8px 0 4px;\'>{title}</div>
                    <div style=\'color:rgba(255,255,255,0.45);font-size:0.78rem;\'>{desc}</div>
                </div>""", unsafe_allow_html=True)
        return

    # Stats bar
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class=\'metric-box\'>
            <div class=\'metric-val\'>{len(st.session_state.articles)}</div>
            <div class=\'metric-lbl\'>Articles</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class=\'metric-box\'>
            <div class=\'metric-val\'>{len(st.session_state.documents)}</div>
            <div class=\'metric-lbl\'>Chunks</div></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class=\'metric-box\'>
            <div class=\'metric-val\'>{len(st.session_state.chat_history)}</div>
            <div class=\'metric-lbl\'>Questions Asked</div></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class=\'metric-box\'>
            <div class=\'metric-val\'>RAG</div>
            <div class=\'metric-lbl\'>Mode</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Question input
    col_q, col_btn = st.columns([5,1])
    with col_q:
        # Check if sample question was clicked
        default_q = st.session_state.get("sample_q","")
        question  = st.text_input(
            "Ask a question about the news",
            placeholder = "e.g. What did RBI announce about interest rates?",
            value       = default_q,
            label_visibility = "collapsed"
        )
        if default_q:
            st.session_state["sample_q"] = ""

    with col_btn:
        ask_btn = st.button("Ask 🤖", use_container_width=True)

    # Run RAG
    if ask_btn and question:
        with st.spinner("🔍 Searching news + generating answer..."):
            result = run_rag_pipeline(
                question        = question,
                groq_client     = st.session_state.groq_client,
                embedding_model = st.session_state.embed_model,
                faiss_index     = st.session_state.faiss_index,
                documents       = st.session_state.documents,
                top_k           = 5
            )

            # Save to chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer"  : result["answer"],
                "sources" : result["sources"],
                "chunks"  : result["chunks"]
            })

    # Show chat history (latest first)
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"""
        <div class=\'card\'>
            <p style=\'color:rgba(255,255,255,0.5);
               font-size:0.78rem;margin-bottom:6px;\'>
               ❓ QUESTION</p>
            <p style=\'font-weight:700;font-size:1rem;
               margin:0;\'>{chat["question"]}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class=\'answer-box\'>
            <p style=\'color:rgba(255,255,255,0.45);
               font-size:0.75rem;margin-bottom:8px;\'>
               💬 AI ANSWER (from real news)</p>
            <p style=\'margin:0;line-height:1.8;\'>{chat["answer"]}</p>
        </div>""", unsafe_allow_html=True)

        # Sources
        if chat["sources"]:
            st.markdown("**📰 Sources Used:**")
            for src in chat["sources"][:3]:
                st.markdown(f"""
                <div class=\'source-card\'>
                    <strong>{src["source"]}</strong>
                    &nbsp;·&nbsp;
                    <span style=\'color:rgba(255,255,255,0.5);
                          font-size:0.82rem;\'>{src["date"]}</span>
                    &nbsp;·&nbsp;
                    <span style=\'color:#818cf8;font-size:0.82rem;\'>
                        Score: {src["score"]}%</span><br>
                    <span style=\'color:rgba(255,255,255,0.65);
                          font-size:0.85rem;\'>{src["title"][:80]}...</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")


# ─────────────────────────────────────────
# PAGE 2 — NLP ANALYSIS
# ─────────────────────────────────────────
def page_nlp():
    st.title("🔍 NLP Analysis")
    st.markdown("*Key entities and topics extracted from the news*")

    if not st.session_state.pipeline_ready:
        st.info("👈 Build pipeline first from sidebar!")
        return

    nlp_model = load_nlp()

    # Extract entities
    with st.spinner("🔍 Extracting entities..."):
        all_entities = extract_from_articles(
            nlp_model, st.session_state.articles
        )
        topics = get_top_topics(
            st.session_state.articles, top_n=15
        )
        formatted = format_entities_display(all_entities)

    # Entities
    st.markdown("### 🏷️ Named Entities")
    st.markdown("*People, organizations, places mentioned in the news*")

    cols = st.columns(2)
    for i, (label, items) in enumerate(formatted.items()):
        with cols[i % 2]:
            st.markdown(f"**{label}**")
            tags = " ".join([
                f"<span class=\'entity-tag\'>{item}</span>"
                for item in items
            ])
            st.markdown(
                f"<div style=\'margin-bottom:16px;\'>{tags}</div>",
                unsafe_allow_html=True
            )

    # Topics
    st.markdown("### 🔥 Trending Topics")
    st.markdown("*Most frequently mentioned keywords*")
    topic_tags = " ".join([
        f"<span class=\'topic-tag\'>{word} ({count}x)</span>"
        for word, count in topics
    ])
    st.markdown(
        f"<div style=\'margin-bottom:16px;\'>{topic_tags}</div>",
        unsafe_allow_html=True
    )

    # Article list
    st.markdown("### 📰 All Fetched Articles")
    for art in st.session_state.articles:
        st.markdown(f"""
        <div class=\'source-card\'>
            <strong>{art["title"][:80]}</strong><br>
            <span style=\'color:rgba(255,255,255,0.45);font-size:0.80rem;\'>
                {art["source"]} &nbsp;·&nbsp; {art["published_at"]}
            </span>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE 3 — CHAT HISTORY
# ─────────────────────────────────────────
def page_history():
    st.title("🗂️ Chat History")

    if not st.session_state.chat_history:
        st.info("💬 No questions asked yet!")
        return

    st.markdown(f"**{len(st.session_state.chat_history)} questions asked**")

    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.rerun()

    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:60]}..."):
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown(f"**Answer:** {chat['answer']}")
            if chat["sources"]:
                st.markdown("**Sources:**")
                for src in chat["sources"]:
                    st.markdown(f"- {src['source']} | {src['date']}")


# ─────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────
def main():
    news_key, groq_key, query, max_articles, days_back, build_btn = render_sidebar()

    # Build pipeline if button clicked
    if build_btn:
        build_pipeline(news_key, groq_key, query, max_articles, days_back)

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["🤖 Ask Questions", "🔍 NLP Analysis", "🗂️ History"],
        label_visibility="collapsed"
    )

    if page == "🤖 Ask Questions":
        page_ask()
    elif page == "🔍 NLP Analysis":
        page_nlp()
    elif page == "🗂️ History":
        page_history()

    st.markdown("""
    <div class=\'footer\'>
        NewsRAG AI · Powered by LangChain + FAISS + Groq + SpaCy
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
'''

with open("app.py","w") as f:
    f.write(code)
print("✅ app.py created!")
