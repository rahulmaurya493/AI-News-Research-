"""
NEWS FETCHER — Phase 1
======================
WHAT  : Fetches real news articles from NewsAPI
WHY   : This is our data source for RAG pipeline
HOW   : Query NewsAPI → parse JSON → clean → return list of articles

Concepts used:
- REST API calls
- JSON parsing
- Data cleaning
- Error handling
"""

from newsapi import NewsApiClient
from datetime import datetime, timedelta
import pandas as pd
import os


# ─────────────────────────────────────────
# WHAT  : Initialize NewsAPI client
# WHY   : We need authenticated connection to fetch news
# HOW   : Pass API key → get client object back
# ─────────────────────────────────────────
def get_news_client(api_key):
    """
    Create and return NewsAPI client.

    Args:
        api_key : your NewsAPI key from newsapi.org

    Returns:
        NewsApiClient object
    """
    client = NewsApiClient(api_key=api_key)
    print("✅ NewsAPI client ready!")
    return client


# ─────────────────────────────────────────
# WHAT  : Fetch articles for a query
# WHY   : Get relevant news based on user's topic
# HOW   : Send query to NewsAPI → get back JSON → parse articles
# ─────────────────────────────────────────
def fetch_articles(client, query, days_back=7, max_articles=20, language="en"):
    """
    Fetch news articles for a given query.

    CONCEPT — Why days_back=7?
    NewsAPI free tier only gives last 30 days.
    7 days gives recent relevant news without too much noise.

    Args:
        client       : NewsAPI client
        query        : search topic e.g. "RBI interest rates"
        days_back    : how many days of news to fetch
        max_articles : max articles to return
        language     : "en" for English

    Returns:
        List of cleaned article dicts
    """
    try:
        # Calculate date range
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        print(f"\n🔍 Searching news for: '{query}'")
        print(f"📅 Date range: {start_date.date()} → {end_date.date()}")

        # ── CONCEPT: API call ──
        # NewsAPI returns JSON with:
        # {status, totalResults, articles: [{source, author, title, description, url, content...}]}
        response = client.get_everything(
            q          = query,
            from_param = start_date.strftime("%Y-%m-%d"),
            to         = end_date.strftime("%Y-%m-%d"),
            language   = language,
            sort_by    = "relevancy",   # most relevant first
            page_size  = max_articles
        )

        if response["status"] != "ok":
            print(f"❌ NewsAPI error: {response.get('message')}")
            return []

        raw_articles = response["articles"]
        print(f"📰 Found {response['totalResults']} total | Fetching top {len(raw_articles)}")

        # ── Clean and structure articles ──
        articles = []
        for i, art in enumerate(raw_articles):

            # CONCEPT: Why clean? Raw API data has None values,
            # "[Removed]" placeholders, very short content
            # We filter these out for better RAG quality

            title       = art.get("title")       or ""
            description = art.get("description") or ""
            content     = art.get("content")     or ""
            url         = art.get("url")          or ""
            source      = art.get("source", {}).get("name") or "Unknown"
            published   = art.get("publishedAt")  or ""

            # Skip removed or empty articles
            if "[Removed]" in title or not title:
                continue
            if len(content) < 50 and len(description) < 50:
                continue

            # ── CONCEPT: Why combine title + description + content? ──
            # Each field alone is incomplete:
            # title       = headline only (short)
            # description = 1-2 lines summary
            # content     = full text (sometimes truncated by NewsAPI)
            # Combined = richer text for embeddings
            full_text = f"""
Title      : {title}
Source     : {source}
Published  : {published[:10] if published else 'Unknown'}
Description: {description}
Content    : {content}
URL        : {url}
""".strip()

            articles.append({
                "id"          : i + 1,
                "title"       : title,
                "source"      : source,
                "published_at": published[:10] if published else "Unknown",
                "description" : description,
                "content"     : content,
                "url"         : url,
                "full_text"   : full_text,
                "query"       : query
            })

        print(f"✅ Cleaned articles ready: {len(articles)}")
        return articles

    except Exception as e:
        print(f"❌ Error fetching news: {e}")
        return []


# ─────────────────────────────────────────
# WHAT  : Fetch top headlines for India
# WHY   : For general news dashboard
# HOW   : Use get_top_headlines endpoint
# ─────────────────────────────────────────
def fetch_top_headlines(client, category="business", country="in", max_articles=10):
    """
    Fetch today's top headlines.

    CONCEPT — Why separate function?
    NewsAPI has 2 endpoints:
    1. get_everything  → search by keyword (used above)
    2. get_top_headlines → today's top news by category/country

    Categories: business, technology, science, health, sports, entertainment

    Args:
        category     : news category
        country      : "in" for India, "us" for USA
        max_articles : max to return
    """
    try:
        print(f"\n📰 Fetching top {category} headlines for India...")

        response = client.get_top_headlines(
            category  = category,
            country   = country,
            page_size = max_articles
        )

        if response["status"] != "ok":
            print(f"❌ Error: {response.get('message')}")
            return []

        articles = []
        for i, art in enumerate(response["articles"]):
            title       = art.get("title")       or ""
            description = art.get("description") or ""
            content     = art.get("content")     or ""
            source      = art.get("source", {}).get("name") or "Unknown"
            published   = art.get("publishedAt") or ""
            url         = art.get("url")         or ""

            if "[Removed]" in title or not title:
                continue

            full_text = f"""
Title      : {title}
Source     : {source}
Published  : {published[:10] if published else 'Unknown'}
Description: {description}
Content    : {content}
URL        : {url}
""".strip()

            articles.append({
                "id"          : i + 1,
                "title"       : title,
                "source"      : source,
                "published_at": published[:10] if published else "Unknown",
                "description" : description,
                "content"     : content,
                "url"         : url,
                "full_text"   : full_text,
                "query"       : f"top {category} headlines"
            })

        print(f"✅ Headlines fetched: {len(articles)}")
        return articles

    except Exception as e:
        print(f"❌ Error fetching headlines: {e}")
        return []


# ─────────────────────────────────────────
# WHAT  : Show summary of fetched articles
# WHY   : Quick check before moving to next phase
# HOW   : Print key info from each article
# ─────────────────────────────────────────
def show_articles_summary(articles):
    """
    Print clean summary of fetched articles.
    Use this to verify data before embedding.
    """
    if not articles:
        print("❌ No articles to show!")
        return

    print(f"\n{'='*55}")
    print(f"  📰 ARTICLES SUMMARY ({len(articles)} articles)")
    print(f"{'='*55}")

    for art in articles:
        print(f"\n  [{art['id']}] {art['title'][:70]}...")
        print(f"      Source  : {art['source']}")
        print(f"      Date    : {art['published_at']}")
        print(f"      Length  : {len(art['full_text'])} characters")
        print(f"      URL     : {art['url'][:60]}...")

    print(f"\n{'='*55}")
    print(f"  Total text: {sum(len(a['full_text']) for a in articles):,} characters")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────
# WHAT  : Save articles to CSV
# WHY   : Cache fetched articles locally
#         so we don't waste API calls during testing
# HOW   : Convert list to DataFrame → save CSV
# ─────────────────────────────────────────
def save_articles(articles, query, folder="data"):
    """
    Save articles to CSV for caching.

    CONCEPT — Why cache?
    NewsAPI free tier has 100 requests/day limit.
    Saving locally means we can test RAG pipeline
    without burning API calls every time!
    """
    os.makedirs(folder, exist_ok=True)

    clean_query = query.replace(" ", "_").replace("/", "_")[:30]
    filename    = f"{folder}/articles_{clean_query}.csv"

    df = pd.DataFrame(articles)
    df.to_csv(filename, index=False)

    print(f"💾 Articles saved: {filename}")
    return filename


# ─────────────────────────────────────────
# WHAT  : Load saved articles from CSV
# WHY   : Reuse cached articles without API call
# ─────────────────────────────────────────
def load_articles(query, folder="data"):
    """Load previously saved articles from CSV."""
    clean_query = query.replace(" ", "_").replace("/", "_")[:30]
    filename    = f"{folder}/articles_{clean_query}.csv"

    if not os.path.exists(filename):
        print(f"⚠️ No cached articles found for '{query}'")
        return None

    df       = pd.read_csv(filename)
    articles = df.to_dict("records")
    print(f"📂 Loaded {len(articles)} cached articles for '{query}'")
    return articles


# ─────────────────────────────────────────
# TEST — Run directly in Jupyter
# ─────────────────────────────────────────
if __name__ == "__main__":

    # ── Replace with your actual key ──
    import streamlit as st

# 1. Pull the key from your Streamlit Dashboard Secrets
# This replaces the literal "2972c7..." string
    API_KEY = st.secrets.get("news_key", "")

# 2. Check if the key exists before connecting
    if not API_KEY:
        st.error("❌ NewsAPI key not found in Streamlit Secrets!")
        st.stop()

# Step 1: Connect
    client = get_news_client(API_KEY)

    # Step 2: Fetch articles
    articles = fetch_articles(
        client,
        query        = "RBI interest rates India",
        days_back    = 7,
        max_articles = 10
    )

    # Step 3: Show summary
    show_articles_summary(articles)

    # Step 4: Save for later use
    if articles:
        save_articles(articles, "RBI interest rates India")

    # Step 5: Test headlines
    headlines = fetch_top_headlines(client, category="business")
    show_articles_summary(headlines)
