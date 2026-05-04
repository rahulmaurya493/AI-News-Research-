
from groq import Groq
import sys, os
sys.path.append("src")

from embeddings   import get_embedding_model, embed_documents, embed_query
from vector_store import build_faiss_index, search_index, save_faiss_index, load_faiss_index

# ─────────────────────────────────────────
# CONCEPT: What is a Prompt Template?
# ─────────────────────────────────────────
# A prompt template is a structured message
# we send to the LLM with:
# 1. Instructions  → how to behave
# 2. Context       → the retrieved news chunks
# 3. Question      → what user asked
#
# WHY structure matters:
# Bad prompt  → vague answer, hallucination
# Good prompt → accurate, cited, focused answer
# ─────────────────────────────────────────

RAG_PROMPT = """You are a helpful AI news research assistant.
Answer the user question ONLY based on the provided news context below.
If the answer is not in the context, say "I could not find relevant information in the current news."

Always mention the source name and date when citing information.
Be concise, clear and factual.

NEWS CONTEXT:
{context}

USER QUESTION:
{question}

YOUR ANSWER:"""


def get_groq_client(api_key):
    """
    WHAT  : Connect to Groq LLM API
    WHY   : Groq gives free, fast LLM inference
    HOW   : Pass API key → get client

    CONCEPT — Why Groq over OpenAI?
    Groq is FREE with generous limits
    Groq uses LPU chips → much faster than GPUs
    Llama3 model → open source, powerful
    """
    client = Groq(api_key=api_key)
    print("✅ Groq client ready!")
    return client


def build_context(retrieved_docs):
    """
    WHAT  : Convert retrieved chunks into readable context
    WHY   : LLM needs clean formatted text as input
    HOW   : Format each chunk with its source info

    CONCEPT — Why include metadata in context?
    If we just send raw chunk text → LLM cant cite sources
    Adding title + source + date → LLM can say
    "According to Economic Times (Apr 1)..."
    """
    context_parts = []

    for i, doc in enumerate(retrieved_docs):
        title  = doc.metadata.get("title", "Unknown")
        source = doc.metadata.get("source", "Unknown")
        date   = doc.metadata.get("published_at", "Unknown")
        score  = doc.metadata.get("similarity_score", 0)
        text   = doc.page_content

        part = f"""
[Article {i+1}]
Title  : {title}
Source : {source}
Date   : {date}
Content: {text}
""".strip()
        context_parts.append(part)

    return "\n\n---\n\n".join(context_parts)


def ask_llm(groq_client, context, question, model="llama-3.1-8b-instant"):
    """
    WHAT  : Send question + context to LLM → get answer
    WHY   : LLM reads context and generates grounded answer
    HOW   : Fill prompt template → send to Groq API

    CONCEPT — LLM Parameters:
    model       = which AI model to use
                  llama3-8b-8192 → fast, good quality
                  llama3-70b-8192 → slower but smarter
    temperature = creativity level
                  0.1 → very focused, factual (best for news)
                  0.9 → creative, varied (best for stories)
    max_tokens  = max length of answer
                  512 → short answer
                  1024 → detailed answer

    Args:
        groq_client : Groq client object
        context     : formatted news chunks as string
        question    : user question
        model       : which LLM to use

    Returns:
        answer string from LLM
    """
    # Fill the prompt template
    prompt = RAG_PROMPT.format(
        context  = context,
        question = question
    )

    print(f"\n🤖 Sending to LLM ({model})...")
    print(f"   Question : {question}")
    print(f"   Context  : {len(context)} characters")

    response = groq_client.chat.completions.create(
        model    = model,
        messages = [
            {
                "role"   : "system",
                "content": "You are a helpful AI news research assistant. Answer only from the provided context."
            },
            {
                "role"   : "user",
                "content": prompt
            }
        ],
        temperature = 0.1,    # low = factual answers
        max_tokens  = 1024,
    )

    answer = response.choices[0].message.content
    print(f"✅ Answer received! ({len(answer)} characters)")
    return answer


def run_rag_pipeline(
    question,
    groq_client,
    embedding_model,
    faiss_index,
    documents,
    top_k = 5
):
    """
    MAIN FUNCTION — Run complete RAG pipeline.

    WHAT  : Full pipeline from question to answer
    WHY   : One function to call everything in order
    HOW   : Embed → Search → Build context → LLM → Return

    Args:
        question        : user question string
        groq_client     : connected Groq client
        embedding_model : loaded SentenceTransformer
        faiss_index     : built FAISS index
        documents       : list of Document objects
        top_k           : chunks to retrieve

    Returns:
        dict with answer + sources + retrieved chunks
    """
    print(f"\n{'='*55}")
    print(f"  🚀 RAG PIPELINE RUNNING")
    print(f"  Question: {question}")
    print(f"{'='*55}")

    # ── STEP 1: Embed the question ──
    print("\n📍 Step 1: Embedding question...")
    query_vector = embed_query(embedding_model, question)

    # ── STEP 2: Search FAISS for relevant chunks ──
    print("\n📍 Step 2: Searching FAISS...")
    retrieved_docs = search_index(
        faiss_index, documents, query_vector, top_k=top_k
    )

    if not retrieved_docs:
        return {
            "answer"  : "No relevant news found for your question.",
            "sources" : [],
            "chunks"  : []
        }

    # ── STEP 3: Build context from chunks ──
    print("\n📍 Step 3: Building context...")
    context = build_context(retrieved_docs)

    # ── STEP 4: Send to LLM ──
    print("\n📍 Step 4: Getting LLM answer...")
    answer = ask_llm(groq_client, context, question)

    # ── STEP 5: Extract unique sources ──
    sources = []
    seen    = set()
    for doc in retrieved_docs:
        source = doc.metadata.get("source","Unknown")
        title  = doc.metadata.get("title","Unknown")
        url    = doc.metadata.get("url","")
        date   = doc.metadata.get("published_at","Unknown")
        key    = f"{source}_{title[:30]}"
        if key not in seen:
            sources.append({
                "source": source,
                "title" : title,
                "url"   : url,
                "date"  : date,
                "score" : doc.metadata.get("similarity_score", 0)
            })
            seen.add(key)

    print(f"\n{'='*55}")
    print(f"  ✅ RAG PIPELINE COMPLETE!")
    print(f"  Sources used: {len(sources)}")
    print(f"{'='*55}")

    return {
        "answer"  : answer,
        "sources" : sources,
        "chunks"  : retrieved_docs
    }


def setup_pipeline(news_api_key, groq_api_key, query, max_articles=15):
    """
    SETUP FUNCTION — Build entire pipeline from scratch.

    Call this once per topic/query.
    Then call run_rag_pipeline() for each question.

    Steps:
    1. Fetch news
    2. Chunk articles
    3. Embed chunks
    4. Build FAISS index
    5. Return everything ready to use
    """
    from news_fetcher import get_news_client, fetch_articles
    from chunker      import chunk_all_articles

    print(f"\n🔧 Setting up RAG pipeline for: {query}")
    print("="*55)

    # Step 1: Fetch news
    print("\n[1/4] Fetching news...")
    news_client = get_news_client(news_api_key)
    articles    = fetch_articles(
        news_client, query=query, max_articles=max_articles
    )

    if not articles:
        print("❌ No articles found!")
        return None, None, None, None

    # Step 2: Chunk
    print("\n[2/4] Chunking articles...")
    documents = chunk_all_articles(articles)

    # Step 3: Embed
    print("\n[3/4] Embedding chunks...")
    embedding_model      = get_embedding_model()
    texts, vectors, docs = embed_documents(embedding_model, documents)

    # Step 4: Build FAISS
    print("\n[4/4] Building FAISS index...")
    faiss_index = build_faiss_index(vectors)
    save_faiss_index(faiss_index, docs, query)

    # Groq client
    groq_client = get_groq_client(groq_api_key)

    print("\n✅ Pipeline ready! Start asking questions.")
    return groq_client, embedding_model, faiss_index, docs
