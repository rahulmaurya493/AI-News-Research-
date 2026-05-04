
import faiss
import numpy as np
import pickle, os

# ─────────────────────────────────────────
# CONCEPT: How FAISS works?
# ─────────────────────────────────────────
# FAISS = Facebook AI Similarity Search
#
# Normal search: checks every vector one by one
# FAISS search : uses smart indexing to find
#                similar vectors INSTANTLY
#                even with millions of vectors!
#
# IndexFlatL2 = simplest FAISS index
# L2 = Euclidean distance between vectors
# Flat = checks all vectors (fine for <10k chunks)
# ─────────────────────────────────────────

def build_faiss_index(vectors):
    """
    Build FAISS index from vectors.

    WHAT  : Creates searchable index from all chunk vectors
    WHY   : Enables fast similarity search
    HOW   : IndexFlatL2 stores vectors + enables search

    Args:
        vectors : numpy array shape (n_chunks, 384)

    Returns:
        FAISS index object
    """
    # CONCEPT: vectors must be float32 for FAISS
    vectors = np.array(vectors).astype("float32")

    dimension = vectors.shape[1]   # 384
    n_vectors = vectors.shape[0]   # 20

    print(f"\n🗄️  Building FAISS index...")
    print(f"   Vectors  : {n_vectors}")
    print(f"   Dimension: {dimension}")

    # Create index
    # IndexFlatL2 = exact search using L2 distance
    index = faiss.IndexFlatL2(dimension)

    # Add all vectors to index
    index.add(vectors)

    print(f"✅ FAISS index built!")
    print(f"   Total vectors stored: {index.ntotal}")
    return index

def search_index(index, documents, query_vector, top_k=5):
    """
    Search FAISS for most similar chunks.

    WHAT  : Finds top_k chunks most similar to question
    WHY   : These chunks are the relevant context for LLM
    HOW   : Compare question vector vs all chunk vectors
            Return closest ones

    CONCEPT — What is top_k?
    top_k = 5 means return 5 most similar chunks
    Too low (1-2) → LLM misses important context
    Too high (10+) → LLM gets confused with too much
    5 is the sweet spot!

    Args:
        index        : FAISS index
        documents    : list of LangChain Documents
        query_vector : embedded question (shape 384,)
        top_k        : how many chunks to return

    Returns:
        list of most relevant Document objects
    """
    # Reshape for FAISS — needs (1, 384) not (384,)
    query_vec = np.array([query_vector]).astype("float32")

    # Search!
    # distances = how far each result is (lower = more similar)
    # indices   = which chunks matched
    distances, indices = index.search(query_vec, top_k)

    print(f"\n🔍 Search Results (top {top_k}):")
    print(f"{'='*55}")

    results = []
    for rank, (dist, idx) in enumerate(
            zip(distances[0], indices[0])):

        if idx == -1:   # FAISS returns -1 if not enough results
            continue

        doc = documents[idx]

        # CONCEPT: Convert L2 distance to similarity score
        # Lower distance = more similar
        # We convert to 0-100 score for display
        similarity = round(float(1 / (1 + dist)) * 100, 1)

        print(f"\n  Rank {rank+1} | Score: {similarity}%")
        print(f"  Title : {doc.metadata.get('title','Unknown')[:50]}...")
        print(f"  Source: {doc.metadata.get('source','Unknown')}")
        print(f"  Chunk : {doc.page_content[:100]}...")

        # Add similarity score to metadata
        doc.metadata["similarity_score"] = similarity
        doc.metadata["rank"]             = rank + 1
        results.append(doc)

    print(f"\n{'='*55}")
    return results

def save_faiss_index(index, documents, query, folder="data/vector_store"):
    """
    Save FAISS index + documents together.

    CONCEPT — Why save both together?
    FAISS index stores vectors but NOT the original text!
    We need documents separately to get text back after search.
    So we always save them as a pair.
    """
    os.makedirs(folder, exist_ok=True)
    clean = query.replace(" ","_")[:30]

    # Save FAISS index
    faiss_path = f"{folder}/index_{clean}.faiss"
    faiss.write_index(index, faiss_path)

    # Save documents separately
    docs_path  = f"{folder}/docs_{clean}.pkl"
    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)

    print(f"💾 FAISS index saved : {faiss_path}")
    print(f"💾 Documents saved   : {docs_path}")
    return faiss_path, docs_path

def load_faiss_index(query, folder="data/vector_store"):
    """Load saved FAISS index + documents."""
    clean      = query.replace(" ","_")[:30]
    faiss_path = f"{folder}/index_{clean}.faiss"
    docs_path  = f"{folder}/docs_{clean}.pkl"

    if not os.path.exists(faiss_path):
        print(f"⚠️  No saved index for \'{query}\'")
        return None, None

    index = faiss.read_index(faiss_path)
    with open(docs_path, "rb") as f:
        documents = pickle.load(f)

    print(f"📂 Loaded FAISS index: {index.ntotal} vectors")
    print(f"📂 Loaded documents  : {len(documents)} chunks")
    return index, documents

def build_and_save(vectors, documents, query):
    """One call to build + save everything."""
    index = build_faiss_index(vectors)
    save_faiss_index(index, documents, query)
    return index
