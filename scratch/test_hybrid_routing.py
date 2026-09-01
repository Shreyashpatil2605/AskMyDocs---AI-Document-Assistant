import os
import sys

# Ensure local imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.documents import Document
from src.embeddings import get_embeddings_model
from src.vector_store import build_vector_store, retrieve_dynamic_documents, retrieve_documents_for_summary
from src.rag_pipeline import is_global_document_query

def test_intent_detection():
    print("Testing global document query intent detection...")
    
    # Test 1: Summarize query
    assert is_global_document_query("Summarize the uploaded documents.") == True, "Failed Test 1"
    print("[OK] Test 1 Passed: 'Summarize the uploaded documents.' detected as global.")
    
    # Test 2: Main topics query
    assert is_global_document_query("What are the main topics?") == True, "Failed Test 2"
    print("[OK] Test 2 Passed: 'What are the main topics?' detected as global.")
    
    # Test 3: Important points query
    assert is_global_document_query("What are the important points?") == True, "Failed Test 3"
    print("[OK] Test 3 Passed: 'What are the important points?' detected as global.")
    
    # Test 4: Specific query
    assert is_global_document_query("What is FAISS?") == False, "Failed Test 4"
    print("[OK] Test 4 Passed: 'What is FAISS?' detected as specific.")
    
    # Additional specific query
    assert is_global_document_query("How to bake a chocolate cake?") == False, "Failed Test 4 (unrelated)"
    print("[OK] Additional Specific Query detected as specific.")
    print("All intent detection tests passed successfully!\n")

def test_hybrid_routing_and_thresholds():
    print("Testing hybrid routing and similarity thresholds...")
    embeddings = get_embeddings_model()
    
    # Create mock chunks
    chunks = [
        Document(page_content="FAISS similarity search uses dense vector distances.", metadata={"source": "doc1.pdf", "page": 1, "chunk_index": 0}),
        Document(page_content="L2 distance flat indexing is fast for small sets.", metadata={"source": "doc1.pdf", "page": 2, "chunk_index": 1}),
        Document(page_content="The sun shines bright in a cloudless summer sky.", metadata={"source": "doc2.pdf", "page": 1, "chunk_index": 0})
    ]
    vector_store = build_vector_store(chunks, embeddings)
    
    # Test 1 Summary Retrieval Flow
    # Must retrieve chunks without failing due to similarity threshold
    docs_summary = retrieve_documents_for_summary(vector_store)
    assert len(docs_summary) > 0, "Summary retrieval failed to return documents"
    print("[OK] Summary retrieval returned chunks successfully.")
    
    # Test 5: Completely unrelated specific question (should return 0 chunks due to similarity threshold < 0.35)
    unrelated_query = "How to bake a chocolate cake?"
    docs_specific = retrieve_dynamic_documents(vector_store, unrelated_query, min_similarity=0.35)
    assert len(docs_specific) == 0, f"Expected 0 chunks for unrelated query, got {len(docs_specific)}"
    print("[OK] Test 5 Passed: Unrelated query returned 0 chunks successfully.")
    print("All routing and threshold tests passed successfully!\n")

def test_sorting_and_budgets():
    print("Testing sorting and budget allocation rules...")
    
    # Scenario A: 2 Documents (1-3 documents -> 3 chunks per document)
    chunks_2_docs = []
    # Document 1 has 5 chunks
    for i in range(5):
        chunks_2_docs.append(Document(
            page_content=f"Doc1 chunk {i}",
            metadata={"source": "doc1.pdf", "page": (i // 2) + 1, "chunk_index": i}
        ))
    # Document 2 has 2 chunks
    for i in range(2):
        chunks_2_docs.append(Document(
            page_content=f"Doc2 chunk {i}",
            metadata={"source": "doc2.pdf", "page": 1, "chunk_index": i}
        ))
        
    embeddings = get_embeddings_model()
    vs = build_vector_store(chunks_2_docs, embeddings)
    retrieved = retrieve_documents_for_summary(vs)
    
    # doc1 should have 3 representative chunks (beginning, middle, end)
    # doc2 should have 2 chunks (since it only has 2 chunks total <= 3 budget)
    # Total retrieved should be 5
    assert len(retrieved) == 5, f"Expected 5 chunks retrieved, got {len(retrieved)}"
    
    # Check that retrieved doc1 chunks are sorted chronologically and represent beginning, middle, end
    doc1_retrieved = [d for d in retrieved if d.metadata["source"] == "doc1.pdf"]
    assert len(doc1_retrieved) == 3, f"Expected 3 chunks for doc1, got {len(doc1_retrieved)}"
    assert doc1_retrieved[0].metadata["chunk_index"] == 0, "Expected beginning chunk"
    assert doc1_retrieved[1].metadata["chunk_index"] == 2, "Expected middle chunk"
    assert doc1_retrieved[2].metadata["chunk_index"] == 4, "Expected end chunk"
    print("[OK] Chronological sorting and Beginning-Middle-End representation verified.")
    
    # Scenario B: 5 Documents (4-9 documents -> 2 chunks per document)
    chunks_5_docs = []
    for doc_id in range(1, 6):
        # 4 chunks per doc
        for i in range(4):
            chunks_5_docs.append(Document(
                page_content=f"Doc{doc_id} chunk {i}",
                metadata={"source": f"doc{doc_id}.pdf", "page": 1, "chunk_index": i}
            ))
            
    vs_5 = build_vector_store(chunks_5_docs, embeddings)
    retrieved_5 = retrieve_documents_for_summary(vs_5)
    # 5 docs * 2 chunks = 10 chunks, capped strictly at 10!
    assert len(retrieved_5) == 10, f"Expected strictly capped 10 chunks, got {len(retrieved_5)}"
    print("[OK] Strict total cap of 10 chunks verified.")
    print("All sorting and budget tests passed successfully!\n")

if __name__ == "__main__":
    try:
        test_intent_detection()
        test_hybrid_routing_and_thresholds()
        test_sorting_and_budgets()
        print("All Hybrid Query Intent Routing tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
