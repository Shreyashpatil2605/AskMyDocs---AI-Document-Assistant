import os
import sys
import hashlib
from typing import List

# Ensure local imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.documents import Document
from src.embeddings import get_embeddings_model
from src.vector_store import build_vector_store

# Mock Streamlit session state as a class for testing
class MockSessionState:
    def __init__(self):
        self.vector_store = None
        self.processed_docs = {}
        self.processed_hashes = []
        self.file_stats = {}
        self.total_pages = 0
        self.total_chunks = 0
        self.is_processed = False
        self.processed_filenames = []

class MockUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self.content = content
    def getvalue(self) -> bytes:
        return self.content

def calculate_sha256(file_bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def test_incremental_processing():
    print("Testing incremental processing logic...")
    state = MockSessionState()
    embeddings_model = get_embeddings_model()
    
    # 1. Simulate Uploading File A
    file_a = MockUploadedFile("DocA.pdf", b"Dummy PDF content for Document A")
    file_a_hash = calculate_sha256(file_a.getvalue())
    
    print(f"Uploading DocA.pdf (hash: {file_a_hash})...")
    
    # Process Doc A
    current_uploads = [(file_a, file_a_hash)]
    new_files_to_process = []
    chunks_to_add = []
    new_processed_hashes = []
    
    for f, f_hash in current_uploads:
        new_processed_hashes.append(f_hash)
        if f_hash not in state.processed_docs:
            new_files_to_process.append((f, f_hash))
            
    assert len(new_files_to_process) == 1
    print("DocA.pdf correctly marked for processing.")
    
    # Simulate extraction and chunking of Doc A
    doc_a_chunks = [
        Document(page_content="Doc A Content Chunk 1", metadata={"source": "DocA.pdf", "page": 1}),
        Document(page_content="Doc A Content Chunk 2", metadata={"source": "DocA.pdf", "page": 2})
    ]
    
    # Cache Doc A
    for f, f_hash in new_files_to_process:
        state.processed_docs[f_hash] = {
            "filename": f.name,
            "chunks": doc_a_chunks,
            "stats": {"pages": 2, "chunks": 2}
        }
        chunks_to_add.extend(doc_a_chunks)
        
    # Build vector store for Doc A
    state.vector_store = build_vector_store(chunks_to_add, embeddings_model)
    state.processed_hashes = new_processed_hashes
    print("DocA.pdf indexed in FAISS.")
    
    # 2. Simulate Uploading File A again (re-upload / duplicate check)
    print("\nSimulating re-upload of DocA.pdf...")
    current_uploads = [(file_a, file_a_hash)]
    new_files_to_process = []
    
    for f, f_hash in current_uploads:
        if f_hash not in state.processed_docs:
            new_files_to_process.append((f, f_hash))
            
    assert len(new_files_to_process) == 0
    print("Duplicate DocA.pdf correctly skipped (extracted from cache).")
    
    # 3. Simulate Uploading File B alongside File A
    file_b = MockUploadedFile("DocB.pdf", b"Dummy PDF content for Document B")
    file_b_hash = calculate_sha256(file_b.getvalue())
    
    print(f"\nUploading DocB.pdf alongside DocA.pdf...")
    current_uploads = [(file_a, file_a_hash), (file_b, file_b_hash)]
    new_files_to_process = []
    chunks_to_add = []
    new_processed_hashes = []
    
    for f, f_hash in current_uploads:
        new_processed_hashes.append(f_hash)
        if f_hash not in state.processed_docs:
            new_files_to_process.append((f, f_hash))
            
    # Only Doc B should be processed
    assert len(new_files_to_process) == 1
    assert new_files_to_process[0][0].name == "DocB.pdf"
    print("Only DocB.pdf marked for processing (DocA.pdf correctly skipped).")
    
    # Simulate extraction and chunking of Doc B
    doc_b_chunks = [
        Document(page_content="Doc B Content Chunk 1", metadata={"source": "DocB.pdf", "page": 1})
    ]
    
    # Cache Doc B
    for f, f_hash in new_files_to_process:
        state.processed_docs[f_hash] = {
            "filename": f.name,
            "chunks": doc_b_chunks,
            "stats": {"pages": 1, "chunks": 1}
        }
        chunks_to_add.extend(doc_b_chunks)
        
    # Incremental update: Add Doc B to existing vector store
    is_superset = set(state.processed_hashes).issubset(set(new_processed_hashes))
    assert is_superset == True
    
    state.vector_store.add_documents(chunks_to_add)
    state.processed_hashes = new_processed_hashes
    print("DocB.pdf successfully added to FAISS incrementally.")
    
    # Verify that similarity search retrieves from both documents
    results = state.vector_store.similarity_search("content", k=3)
    assert len(results) == 3
    sources = [doc.metadata["source"] for doc in results]
    assert "DocA.pdf" in sources
    assert "DocB.pdf" in sources
    print("Similarity search correctly retrieved chunks from both DocA.pdf and DocB.pdf!")

if __name__ == "__main__":
    try:
        test_incremental_processing()
        print("\nAll incremental caching tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
