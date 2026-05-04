"""
TEXT CHUNKER — Phase 2
======================
WHAT  : Splits long articles into small overlapping chunks
WHY   : LLMs have token limits — can't read full articles
        Small chunks = faster search + more accurate answers
HOW   : LangChain RecursiveCharacterTextSplitter

Concepts used:
- Text splitting strategies
- Chunk size vs overlap tradeoff
- LangChain Document objects
- Metadata preservation
"""

# NEW - correct imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd
import pickle
import os


# ─────────────────────────────────────────
# CONCEPT: What is RecursiveCharacterTextSplitter?
# ─────────────────────────────────────────
# LangChain's smartest splitter.
# Tries to split by: paragraph → sentence → word → character
# So it NEVER cuts a sentence in the middle if possible!
#
# Example:
# chunk_size    = 500  → each chunk ~500 characters
# chunk_overlap = 100  → last 100 chars of chunk1 = first 100 of chunk2
#
# WHY overlap?
# Without overlap:
#   Chunk1: "RBI raised rates. This decision was made because..."
#   Chunk2: "inflation was rising." ← loses context!
#
# With overlap:
#   Chunk1: "RBI raised rates. This decision was made because..."
#   Chunk2: "was made because inflation was rising." ← context preserved!
# ─────────────────────────────────────────


def get_text_splitter(chunk_size=500, chunk_overlap=100):
    """
    Create and return a text splitter.

    WHAT  : Creates LangChain text splitter object
    WHY   : Reusable across all articles
    HOW   : RecursiveCharacterTextSplitter tries multiple
            separators in order: paragraph, sentence, word

    Args:
        chunk_size    : max characters per chunk (default 500)
        chunk_overlap : overlap between chunks   (default 100)

    Returns:
        RecursiveCharacterTextSplitter object
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size         = chunk_size,
        chunk_overlap      = chunk_overlap,
        length_function    = len,

        # CONCEPT: separators tried in ORDER
        # First tries \n\n (paragraph break)
        # If chunk still too big → tries \n (line break)
        # Still too big → tries " " (space/word)
        # Still too big → splits by character
        separators         = ["\n\n", "\n", " ", ""]
    )

    print(f"✅ Text splitter ready!")
    print(f"   Chunk size    : {chunk_size} characters")
    print(f"   Chunk overlap : {chunk_overlap} characters")
    return splitter


# ─────────────────────────────────────────
# WHAT  : Convert one article → list of chunks
# WHY   : Each chunk becomes a searchable unit in FAISS
# HOW   : Split full_text → wrap in LangChain Document
# ─────────────────────────────────────────
def chunk_article(article, splitter):
    """
    Split one article into multiple chunks.

    CONCEPT — LangChain Document:
    Document is a standard object with 2 parts:
    1. page_content : the actual text chunk
    2. metadata     : extra info about where chunk came from
                      (title, source, url, date)

    WHY metadata?
    When LLM answers, it can cite:
    "According to [source] on [date]: [answer]"
    Without metadata → no source citation!

    Args:
        article  : dict with full_text, title, source, url etc.
        splitter : RecursiveCharacterTextSplitter object

    Returns:
        List of LangChain Document objects
    """
    full_text = article.get("full_text", "")

    if not full_text or len(full_text) < 50:
        return []

    # Split into raw text chunks
    raw_chunks = splitter.split_text(full_text)

    # ── Wrap each chunk in LangChain Document ──
    # CONCEPT: Document = chunk text + metadata
    # Metadata travels with chunk everywhere
    # FAISS stores it, retriever returns it, LLM sees it
    documents = []
    for i, chunk in enumerate(raw_chunks):
        doc = Document(
            page_content = chunk,
            metadata     = {
                "article_id"  : article.get("id", 0),
                "chunk_id"    : i + 1,
                "title"       : article.get("title", "Unknown"),
                "source"      : article.get("source", "Unknown"),
                "published_at": article.get("published_at", "Unknown"),
                "url"         : article.get("url", ""),
                "query"       : article.get("query", ""),
                "total_chunks": len(raw_chunks)
            }
        )
        documents.append(doc)

    return documents


# ─────────────────────────────────────────
# WHAT  : Chunk ALL articles at once
# WHY   : Process entire news batch in one call
# HOW   : Loop through articles → chunk each → combine
# ─────────────────────────────────────────
def chunk_all_articles(articles, chunk_size=500, chunk_overlap=100):
    """
    Chunk all fetched articles into Documents.

    CONCEPT — Why process all together?
    FAISS vector store needs ALL documents at once
    to build its search index efficiently.

    Args:
        articles      : list of article dicts from news_fetcher
        chunk_size    : characters per chunk
        chunk_overlap : overlap between chunks

    Returns:
        List of all LangChain Document objects
    """
    if not articles:
        print("❌ No articles to chunk!")
        return []

    splitter   = get_text_splitter(chunk_size, chunk_overlap)
    all_docs   = []
    skipped    = 0

    print(f"\n✂️  Chunking {len(articles)} articles...")

    for article in articles:
        docs = chunk_article(article, splitter)
        if docs:
            all_docs.extend(docs)
        else:
            skipped += 1

    print(f"\n✅ Chunking complete!")
    print(f"   Articles processed : {len(articles) - skipped}")
    print(f"   Articles skipped   : {skipped}")
    print(f"   Total chunks made  : {len(all_docs)}")

    if all_docs:
        avg_len = sum(len(d.page_content) for d in all_docs) / len(all_docs)
        print(f"   Avg chunk length   : {avg_len:.0f} characters")

    return all_docs


# ─────────────────────────────────────────
# WHAT  : Show chunk details
# WHY   : Verify chunks look correct before embedding
# ─────────────────────────────────────────
def show_chunks_summary(documents):
    """
    Print summary of all chunks.
    Always verify chunks before sending to FAISS!
    """
    if not documents:
        print("❌ No documents to show!")
        return

    print(f"\n{'='*55}")
    print(f"  ✂️  CHUNKS SUMMARY ({len(documents)} total chunks)")
    print(f"{'='*55}")

    # Group by article
    seen_articles = {}
    for doc in documents:
        title = doc.metadata.get("title", "Unknown")[:45]
        if title not in seen_articles:
            seen_articles[title] = 0
        seen_articles[title] += 1

    for title, count in seen_articles.items():
        print(f"\n  📰 {title}...")
        print(f"     Chunks : {count}")

    print(f"\n  Sample chunk text:")
    print(f"  {'─'*45}")
    sample = documents[0].page_content[:300]
    print(f"  {sample}...")
    print(f"\n  Sample metadata:")
    for k, v in documents[0].metadata.items():
        print(f"    {k:15}: {v}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────
# WHAT  : Save chunks to disk
# WHY   : Avoid re-chunking during testing
# HOW   : Pickle list of Document objects
# ─────────────────────────────────────────
def save_chunks(documents, query, folder="data"):
    """
    Save chunks to disk with pickle.

    CONCEPT — Why pickle?
    LangChain Document objects can't be saved as CSV
    (they have complex structure).
    Pickle serializes Python objects directly to binary file.
    """
    os.makedirs(folder, exist_ok=True)

    clean_query = query.replace(" ", "_").replace("/","_")[:30]
    filename    = f"{folder}/chunks_{clean_query}.pkl"

    with open(filename, "wb") as f:
        pickle.dump(documents, f)

    print(f"💾 Chunks saved: {filename}")
    print(f"   Total chunks : {len(documents)}")
    return filename


# ─────────────────────────────────────────
# WHAT  : Load saved chunks
# WHY   : Skip re-chunking, go straight to embedding
# ─────────────────────────────────────────
def load_chunks(query, folder="data"):
    """Load previously saved chunks."""
    clean_query = query.replace(" ", "_").replace("/","_")[:30]
    filename    = f"{folder}/chunks_{clean_query}.pkl"

    if not os.path.exists(filename):
        print(f"⚠️  No cached chunks for '{query}'")
        return None

    with open(filename, "rb") as f:
        documents = pickle.load(f)

    print(f"📂 Loaded {len(documents)} cached chunks for '{query}'")
    return documents


# ─────────────────────────────────────────
# CONCEPT EXPLAINER — Chunk size guide
# ─────────────────────────────────────────
def explain_chunk_sizes():
    """
    Helper to understand chunk size tradeoffs.
    Run this to understand before choosing chunk_size!
    """
    print("""
    📚 CHUNK SIZE GUIDE
    ═══════════════════════════════════════

    chunk_size = 200  (very small)
    ✅ Very precise retrieval
    ✅ Fast embedding
    ❌ May lose context within a topic
    Use for: short social media text

    chunk_size = 500  (recommended ✅)
    ✅ Good balance of precision + context
    ✅ Works well for news articles
    Use for: news, blogs, articles

    chunk_size = 1000 (large)
    ✅ More context per chunk
    ❌ Less precise retrieval
    ❌ Slower + more expensive
    Use for: books, research papers

    chunk_overlap = 100 (recommended ✅)
    Rule of thumb: overlap = chunk_size / 5
    Too small → context breaks at boundaries
    Too large  → redundant data, wastes memory
    ═══════════════════════════════════════
    """)


# ─────────────────────────────────────────
# TEST — Run directly in Jupyter
# ─────────────────────────────────────────
if __name__ == "__main__":

    import sys
    sys.path.append("src")
    from news_fetcher import get_news_client, fetch_articles, save_articles

    API_KEY  = "2972c76c12ad48e4b4f49c7209494934"
    QUERY    = "RBI interest rates India"

    # Step 1: Fetch articles
    client   = get_news_client(API_KEY)
    articles = fetch_articles(client, query=QUERY, max_articles=10)

    # Step 2: Chunk them
    documents = chunk_all_articles(articles, chunk_size=500, chunk_overlap=100)

    # Step 3: See what we got
    show_chunks_summary(documents)

    # Step 4: Save for Phase 3
    save_chunks(documents, QUERY)

    # Step 5: Understand chunk sizes
    explain_chunk_sizes()

    print("\n🎉 Phase 2 Complete! Ready for Embeddings →")
