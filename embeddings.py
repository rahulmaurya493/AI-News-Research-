
from sentence_transformers  import SentenceTransformer
from langchain_core.documents import Document
import numpy as np
import pickle, os, time

# ─────────────────────────────────────────
# CONCEPT: Which embedding model to use?
# ─────────────────────────────────────────
# all-MiniLM-L6-v2 (recommended ✅)
# → Free, fast, 384-dimensional vectors
# → Great for semantic search
# → Only 80MB download
#
# How it works:
# Model was trained on millions of sentences
# It learned: similar meaning = similar numbers
# So "RBI" and "central bank" get similar vectors!
# ─────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model():
    """
    Load HuggingFace embedding model.

    WHAT  : Downloads + loads sentence transformer model
    WHY   : Converts text to 384-dimensional vectors
    HOW   : SentenceTransformer from HuggingFace
    """
    print(f"⏳ Loading embedding model: {MODEL_NAME}")
    print(f"   (First time = downloads ~80MB, then cached)")
    model = SentenceTransformer(MODEL_NAME)
    print(f"✅ Model loaded!")
    print(f"   Vector size: {model.get_sentence_embedding_dimension()} dimensions")
    return model

def embed_texts(model, texts, batch_size=32, show_progress=True):
    """
    Convert list of texts to vectors.

    CONCEPT — Why batch_size=32?
    Processing all chunks at once = memory error
    Processing one by one = very slow
    Batches of 32 = fast + memory efficient!

    Args:
        model         : SentenceTransformer model
        texts         : list of strings to embed
        batch_size    : how many to process at once
        show_progress : show progress bar

    Returns:
        numpy array of shape (n_texts, 384)
    """
    print(f"\n🔢 Embedding {len(texts)} texts...")
    start = time.time()

    vectors = model.encode(
        texts,
        batch_size      = batch_size,
        show_progress_bar = show_progress,
        convert_to_numpy  = True
    )

    elapsed = time.time() - start
    print(f"✅ Embedding done!")
    print(f"   Texts embedded : {len(texts)}")
    print(f"   Vector shape   : {vectors.shape}")
    print(f"   Time taken     : {elapsed:.1f} seconds")
    return vectors

def embed_documents(model, documents):
    """
    Embed list of LangChain Documents.

    WHAT  : Converts all chunks to vectors
    WHY   : FAISS needs vectors not text
    HOW   : Extract page_content → embed → return

    Args:
        model     : SentenceTransformer model
        documents : list of LangChain Document objects

    Returns:
        tuple of (texts, vectors, documents)
    """
    if not documents:
        print("❌ No documents to embed!")
        return [], None, []

    # Extract text from each Document
    texts   = [doc.page_content for doc in documents]
    vectors = embed_texts(model, texts)
    return texts, vectors, documents

def embed_query(model, query):
    """
    Embed a single user question.

    CONCEPT — Why embed the question?
    FAISS compares vectors to find similar ones.
    Question must be in same vector space as chunks!

    Question vector → compared to → all chunk vectors
    Most similar chunk vectors → most relevant chunks!

    Args:
        query : user question string

    Returns:
        vector of shape (384,)
    """
    vector = model.encode([query], convert_to_numpy=True)[0]
    print(f"✅ Query embedded! Vector size: {len(vector)}")
    return vector

def test_similarity(model):
    """
    Visual demo of how embeddings capture meaning.
    Run this to understand embeddings intuitively!
    """
    from sklearn.metrics.pairwise import cosine_similarity

    sentences = [
        "RBI raised interest rates in India",
        "Central bank increased repo rate",
        "India cricket team won the match",
        "Stock market fell due to rate hike",
        "RBI monetary policy announcement"
    ]

    print("\n🧪 SIMILARITY TEST")
    print("="*55)
    print("Base: \'RBI raised interest rates in India\'")
    print("="*55)

    base_vec = model.encode([sentences[0]])
    for sent in sentences[1:]:
        vec  = model.encode([sent])
        sim  = float(cosine_similarity(base_vec, vec)[0][0])
        bar  = "█" * int(sim * 20)
        print(f"\n  {sent[:45]}")
        print(f"  Similarity: {sim:.3f} {bar}")

    print("="*55)
    print("Higher = more similar meaning!")

def save_vectors(vectors, documents, query, folder="data"):
    """Save vectors + documents together."""
    os.makedirs(folder, exist_ok=True)
    clean = query.replace(" ","_")[:30]
    path  = f"{folder}/vectors_{clean}.pkl"
    with open(path,"wb") as f:
        pickle.dump({"vectors": vectors, "documents": documents}, f)
    print(f"💾 Vectors saved: {path}")
    return path

def load_vectors(query, folder="data"):
    """Load saved vectors."""
    clean = query.replace(" ","_")[:30]
    path  = f"{folder}/vectors_{clean}.pkl"
    if not os.path.exists(path):
        return None, None
    with open(path,"rb") as f:
        data = pickle.load(f)
    print(f"📂 Loaded vectors: {data['vectors'].shape}")
    return data["vectors"], data["documents"]
