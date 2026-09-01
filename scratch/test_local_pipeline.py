import os
import sys

# Ensure local imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings import get_embeddings_model
from src.rag_pipeline import needs_condensation, format_chat_history, resolve_query_locally

def test_embeddings():
    print("Testing local SentenceTransformer embeddings...")
    embeddings = get_embeddings_model()
    test_texts = ["What is FAISS?", "Explain vector embeddings."]
    
    doc_embs = embeddings.embed_documents(test_texts)
    assert len(doc_embs) == 2
    assert len(doc_embs[0]) == 384
    print(f"Document embedding success! Dimensions: {len(doc_embs[0])}")
    
    query_emb = embeddings.embed_query("What is FAISS?")
    assert len(query_emb) == 384
    print("Query embedding success!")

def test_heuristic():
    print("\nTesting query condensation heuristic...")
    chat_history = [
        {"role": "user", "content": "What is FAISS?"},
        {"role": "assistant", "content": "FAISS is a library for similarity search."}
    ]
    
    # Standalone queries (must return False)
    standalone_queries = [
        "What is FAISS?",
        "Explain FAISS.",
        "Can you explain RAG?",
        "How does vector search work?",
        "What are embeddings?",
        "Explain the FAISS index.",
        "Can you explain FAISS?"
    ]
    
    for q in standalone_queries:
        assert not needs_condensation(q, chat_history), f"Failed standalone test for: {q}"
        
    # Context-dependent queries (must return True)
    relative_queries = [
        "How does it work?",
        "What about that?",
        "Explain it.",
        "Where is it stored?",
        "Why is this important?",
        "Tell me more about it.",
        "Why are they important?"
    ]
    
    for q in relative_queries:
        assert needs_condensation(q, chat_history), f"Failed relative test for: {q}"
        
    print("Query condensation heuristic success (all tests passed)!")

def test_local_resolution():
    print("\nTesting local query resolution...")
    chat_history = [
        {"role": "user", "content": "What is FAISS?"},
        {"role": "assistant", "content": "FAISS is a library for similarity search."}
    ]
    
    # Standalone should remain unchanged
    q1 = "Can you explain FAISS?"
    if needs_condensation(q1, chat_history):
        resolved1 = resolve_query_locally(q1, chat_history)
    else:
        resolved1 = q1
    assert resolved1 == q1, f"Expected {q1}, got {resolved1}"
    
    # Context dependent should resolve combined
    q2 = "How does it work?"
    if needs_condensation(q2, chat_history):
        resolved2 = resolve_query_locally(q2, chat_history)
    else:
        resolved2 = q2
    expected2 = "What is FAISS? How does it work?"
    assert resolved2 == expected2, f"Expected {expected2}, got {resolved2}"
    
    print("Local query resolution success!")

def test_history_formatting():
    print("\nTesting chat history formatting...")
    chat_history = [
        {"role": "user", "content": "Turn 1 User"},
        {"role": "assistant", "content": "Turn 1 Assistant"},
        {"role": "user", "content": "Turn 2 User"},
        {"role": "assistant", "content": "Turn 2 Assistant"},
        {"role": "user", "content": "Turn 3 User"},
        {"role": "assistant", "content": "Turn 3 Assistant"}
    ]
    formatted = format_chat_history(chat_history, max_turns=2)
    
    assert "Turn 1" not in formatted
    assert "Turn 2" in formatted
    assert "Turn 3" in formatted
    print("Chat history formatting success!")

if __name__ == "__main__":
    try:
        test_embeddings()
        test_heuristic()
        test_local_resolution()
        test_history_formatting()
        print("\nAll local pipeline tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
