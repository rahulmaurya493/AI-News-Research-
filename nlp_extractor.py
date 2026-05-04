
import spacy
from collections import Counter

# ─────────────────────────────────────────
# CONCEPT: What is NER?
# ─────────────────────────────────────────
# NER = Named Entity Recognition
# SpaCy reads text and labels important words:
#
# "RBI Governor Shaktikanta Das announced
#  a 25bps rate hike in Mumbai on April 5"
#
# Labels:
# RBI               → ORG   (organization)
# Shaktikanta Das   → PERSON
# Mumbai            → GPE   (city/country)
# 25bps             → PERCENT
# April 5           → DATE
# ─────────────────────────────────────────

# Entity labels we care about
ENTITY_LABELS = {
    "ORG"    : "🏢 Organizations",
    "PERSON" : "👤 People",
    "GPE"    : "📍 Locations",
    "MONEY"  : "💰 Money",
    "PERCENT": "📊 Percentages",
    "DATE"   : "📅 Dates",
    "EVENT"  : "🎯 Events",
    "PRODUCT": "📦 Products"
}


def load_nlp_model():
    """
    WHAT  : Load SpaCy English model
    WHY   : We need trained NLP model to find entities
    HOW   : en_core_web_sm = small English model
            Trained on news + web text — perfect for us!
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ SpaCy model loaded!")
        return nlp
    except:
        print("⏳ Downloading SpaCy model...")
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy",
                       "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
        print("✅ SpaCy model ready!")
        return nlp


def extract_entities(nlp, text):
    """
    WHAT  : Find all named entities in text
    WHY   : Shows user key facts at a glance
    HOW   : SpaCy processes text → returns ents

    Args:
        nlp  : loaded SpaCy model
        text : any string to analyze

    Returns:
        dict of {label: [entity names]}
    """
    doc      = nlp(text)
    entities = {}

    for ent in doc.ents:
        label = ent.label_
        text_ = ent.text.strip()

        # Only keep labels we care about
        if label not in ENTITY_LABELS:
            continue

        # Skip very short or numeric-only entities
        if len(text_) < 2:
            continue

        if label not in entities:
            entities[label] = []

        if text_ not in entities[label]:
            entities[label].append(text_)

    return entities


def extract_from_articles(nlp, articles):
    """
    WHAT  : Extract entities from ALL articles at once
    WHY   : Get complete picture of who/what/where
    HOW   : Combine all article texts → run NER

    Returns:
        dict of {label: Counter of most common entities}
    """
    print(f"\n🔍 Extracting entities from {len(articles)} articles...")

    all_entities = {}

    for article in articles:
        text     = article.get("full_text", "")
        entities = extract_entities(nlp, text)

        for label, values in entities.items():
            if label not in all_entities:
                all_entities[label] = []
            all_entities[label].extend(values)

    # Count frequency of each entity
    result = {}
    for label, values in all_entities.items():
        counter = Counter(values)
        # Return top 10 most mentioned
        result[label] = counter.most_common(10)

    print(f"✅ Entity extraction complete!")
    for label, items in result.items():
        display = ENTITY_LABELS.get(label, label)
        print(f"   {display}: {len(items)} unique entities")

    return result


def extract_from_answer(nlp, answer_text):
    """
    WHAT  : Extract entities from LLM answer
    WHY   : Highlight key facts in the answer for user
    HOW   : Same NER on answer text

    Returns:
        dict of {label: [entity names]}
    """
    entities = extract_entities(nlp, answer_text)
    return entities


def get_top_topics(articles, top_n=10):
    """
    WHAT  : Find most mentioned keywords across articles
    WHY   : Shows trending topics in the news
    HOW   : Simple word frequency (excluding stopwords)

    CONCEPT — Why not use SpaCy here?
    For topic frequency, simple counting works fine.
    SpaCy NER is overkill for word frequency.
    Right tool for right job!
    """
    import re

    # Common words to ignore
    stopwords = {
        "the","a","an","in","on","at","to","for","of",
        "and","or","but","is","are","was","were","be",
        "been","being","have","has","had","do","does",
        "did","will","would","could","should","may",
        "might","shall","can","this","that","these",
        "those","it","its","with","from","by","as",
        "not","no","so","if","said","says","also",
        "about","after","before","between","during",
        "into","through","over","under","up","down"
    }

    word_counts = Counter()

    for article in articles:
        text  = article.get("title","") + " " + article.get("description","")
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        for w in words:
            if w not in stopwords:
                word_counts[w] += 1

    return word_counts.most_common(top_n)


def format_entities_display(entities):
    """
    WHAT  : Format entities for clean display
    WHY   : Used in Streamlit app to show entities nicely
    HOW   : Convert to readable dict

    Returns:
        formatted dict for display
    """
    display = {}
    for label, items in entities.items():
        emoji_label = ENTITY_LABELS.get(label, label)
        if isinstance(items[0], tuple):
            # From extract_from_articles (Counter tuples)
            display[emoji_label] = [
                f"{name} ({count}x)"
                for name, count in items[:5]
            ]
        else:
            # From extract_from_answer (plain list)
            display[emoji_label] = items[:5]
    return display
