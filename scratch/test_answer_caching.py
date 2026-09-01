import os
import sys
import hashlib
import re

# Ensure local imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.documents import Document
from src.embeddings import get_embeddings_model
from src.vector_store import build_vector_store, retrieve_dynamic_documents

# Mock Streamlit session state wrapper for test purposes
class MockSessionState:
    def __init__(self):
        self.processed_hashes = []
        self.answer_cache = {}

def normalize_for_cache(q: str) -> str:
    q_clean = q.lower().strip()
    q_clean = re.sub(r"[?.,!]", "", q_clean)
    return " ".join(q_clean.split())

def make_cache_key(processed_hashes, user_query: str, retrieval_query: str) -> str:
    sorted_hashes = sorted(processed_hashes)
    cache_payload = (
        "|".join(sorted_hashes)
        + "|"
        + normalize_for_cache(user_query)
        + "|"
        + normalize_for_cache(retrieval_query)
    )
    return hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()

def test_normalization():
    print("Testing query normalization...")
    q1 = "  What is FAISS? "
    q2 = "what is faiss"
    q3 = "What, is FAISS!"
    assert normalize_for_cache(q1) == "what is faiss"
    assert normalize_for_cache(q2) == "what is faiss"
    assert normalize_for_cache(q3) == "what is faiss"
    print("Query normalization verified successfully!")

def test_stable_cache_keys():
    print("\nTesting document-aware stable cache key generation...")
    hashes_v1 = ["hashA", "hashB"]
    hashes_v2 = ["hashB", "hashA"] # out of order
    hashes_v3 = ["hashA"] # different docs
    
    # Order independence verification
    key1 = make_cache_key(hashes_v1, "What is FAISS?", "What is FAISS?")
    key2 = make_cache_key(hashes_v2, "what is faiss", "what is faiss")
    assert key1 == key2, "Cache key is order sensitive!"
    
    # Document dependency verification
    key3 = make_cache_key(hashes_v3, "What is FAISS?", "What is FAISS?")
    assert key1 != key3, "Cache key did not change when documents changed!"
    
    # Query dependency verification
    key4 = make_cache_key(hashes_v1, "What is FAISS?", "How does it work?")
    assert key1 != key4, "Cache key did not distinguish user vs retrieval queries!"
    print("Stable cache keys verified successfully!")

def test_cache_invalidation_sim():
    print("\nTesting cache invalidation logic...")
    state = MockSessionState()
    state.processed_hashes = ["hashA"]
    state.answer_cache["key1"] = {"answer": "FAISS is great", "sources": []}
    
    # Scenario A: User processes the same documents (no change)
    new_hashes_same = ["hashA"]
    if set(state.processed_hashes) != set(new_hashes_same):
        state.answer_cache = {}
    assert len(state.answer_cache) == 1, "Cache incorrectly invalidated on identical document processing!"
    
    # Scenario B: User processes a new/updated document (change)
    new_hashes_diff = ["hashA", "hashB"]
    if set(state.processed_hashes) != set(new_hashes_diff):
        state.answer_cache = {}
    assert len(state.answer_cache) == 0, "Cache failed to invalidate when document hashes changed!"
    print("Cache invalidation simulation successful!")

def test_dynamic_retrieval_and_thresholds():
    print("\nTesting dynamic retrieve_dynamic_documents thresholds...")
    embeddings = get_embeddings_model()
    chunks = [
        Document(page_content="FAISS similarity search uses dense vector distances.", metadata={"source": "doc1"}),
        Document(page_content="L2 distance flat indexing is fast for small sets.", metadata={"source": "doc2"}),
        Document(page_content="The sun shines bright in a cloudless summer sky.", metadata={"source": "doc3"})
    ]
    vector_store = build_vector_store(chunks, embeddings)
    
    # Let's perform queries and check that the returned list lengths match the logic:
    # score > 0.60 -> 1 chunk
    # score > 0.45 -> 2 chunks
    # otherwise (score >= 0.35) -> 3 chunks (or total chunks in index if < 3)
    # score < 0.35 -> 0 chunks
    
    queries = [
        "L2 distance flat indexing",
        "How to search using FAISS vectors?",
        "How to build a deep neural network?",
        "How to bake a chocolate cake?"
    ]
    
    for q in queries:
        docs = retrieve_dynamic_documents(vector_store, q, min_similarity=0.35)
        
        # Verify with manual score calculations to ensure the function acted correctly
        docs_with_scores = vector_store.similarity_search_with_score(q, k=3)
        if not docs_with_scores:
            assert len(docs) == 0
            continue
            
        first_dist = docs_with_scores[0][1]
        best_score = 1.0 - (first_dist ** 2) / 2.0
        best_score = max(0.0, min(1.0, best_score))
        
        print(f"Query: '{q}' | Best Score: {best_score:.4f} | Retrieved Count: {len(docs)}")
        
        if best_score < 0.35:
            assert len(docs) == 0, f"Expected 0 chunks, got {len(docs)}"
        elif best_score > 0.60:
            assert len(docs) == 1, f"Expected 1 chunk, got {len(docs)}"
        elif best_score > 0.45:
            assert len(docs) == 2, f"Expected 2 chunks, got {len(docs)}"
        else:
            assert len(docs) == min(3, len(chunks)), f"Expected {min(3, len(chunks))} chunks, got {len(docs)}"
            
    print("Dynamic retrieval and similarity thresholds verified successfully!")

if __name__ == "__main__":
    try:
        test_normalization()
        test_stable_cache_keys()
        test_cache_invalidation_sim()
        test_dynamic_retrieval_and_thresholds()
        print("\nAll caching and dynamic retrieval tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
