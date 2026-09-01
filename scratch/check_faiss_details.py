import os
import sys

# Ensure local imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.documents import Document
from src.embeddings import get_embeddings_model
from src.vector_store import build_vector_store

def check_faiss():
    print("Initializing embeddings model...")
    embeddings = get_embeddings_model()
    
    # 1. Check if HuggingFace embeddings are normalized
    # Let's check the L2 norm of an embedded query
    query = "What is FAISS?"
    query_vector = embeddings.embed_query(query)
    l2_norm = sum(x**2 for x in query_vector) ** 0.5
    print(f"L2 Norm of query embedding: {l2_norm:.6f}")
    
    # 2. Build a vector store
    chunks = [
        Document(page_content="FAISS is a library for efficient similarity search.", metadata={"source": "doc1"}),
        Document(page_content="RAG combines retrieval with LLM answer generation.", metadata={"source": "doc2"}),
        Document(page_content="The weather today is sunny and warm.", metadata={"source": "doc3"})
    ]
    vector_store = build_vector_store(chunks, embeddings)
    
    # 3. Check FAISS index details
    print(f"FAISS index type: {type(vector_store.index)}")
    print(f"FAISS metric type: {vector_store.index.metric_type}")
    
    # 4. Search and inspect the exact scores
    print("\nQuery: 'What is FAISS?'")
    results_with_score = vector_store.similarity_search_with_score("What is FAISS?", k=3)
    for doc, score in results_with_score:
        print(f"Content: '{doc.page_content}' | Score (distance): {score:.6f}")
        
    print("\nQuery: 'What is RAG?'")
    results_with_score = vector_store.similarity_search_with_score("What is RAG?", k=3)
    for doc, score in results_with_score:
        print(f"Content: '{doc.page_content}' | Score (distance): {score:.6f}")
        
    print("\nQuery: 'How is the weather?'")
    results_with_score = vector_store.similarity_search_with_score("How is the weather?", k=3)
    for doc, score in results_with_score:
        print(f"Content: '{doc.page_content}' | Score (distance): {score:.6f}")

if __name__ == "__main__":
    check_faiss()
