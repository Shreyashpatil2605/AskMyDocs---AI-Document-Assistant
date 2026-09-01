import os
import sys

# Ensure local imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.documents import Document
from src.embeddings import get_embeddings_model
from src.vector_store import build_vector_store, retrieve_documents_for_summary
from src.prompts import RAG_PROMPT_TEMPLATE

def test_prompt_rules():
    print("Testing prompt template rules...")
    assert "comprehensive" in RAG_PROMPT_TEMPLATE.lower()
    assert "artificially condense" in RAG_PROMPT_TEMPLATE.lower()
    print("[OK] Prompt template rules verified successfully!")

def test_single_doc_chunk_budget():
    print("Testing single-document summary chunk budget expansion...")
    embeddings = get_embeddings_model()
    
    # Single document with 10 chunks
    chunks = []
    for i in range(10):
        chunks.append(Document(
            page_content=f"Paragraph {i} content describing key topic {i}.",
            metadata={"source": "single_doc.pdf", "page": i + 1, "chunk_index": i}
        ))
        
    vs = build_vector_store(chunks, embeddings)
    retrieved = retrieve_documents_for_summary(vs)
    
    # Single document budget should now retrieve 6 representative chunks (instead of capping at 3)
    assert len(retrieved) == 6, f"Expected 6 representative chunks for single document, got {len(retrieved)}"
    
    indices = [d.metadata["chunk_index"] for d in retrieved]
    print(f"Retrieved chunk indices for single 10-chunk document: {indices}")
    assert indices == [0, 2, 4, 6, 8, 9], f"Indices did not match expected spacing: {indices}"
    print("[OK] Single document budget expansion verified successfully!")

if __name__ == "__main__":
    try:
        test_prompt_rules()
        test_single_doc_chunk_budget()
        print("\nAll truncation fix tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
