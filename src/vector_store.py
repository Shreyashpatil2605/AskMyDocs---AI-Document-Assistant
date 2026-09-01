from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Interview Tip: Why FAISS & Similarity Search?
# 1. Why FAISS? FAISS (Facebook AI Similarity Search) is an open-source library optimized for fast,
#    efficient similarity search and clustering of dense vectors. It is much faster than comparing 
#    a query against every single document chunk sequentially (which would be O(N)).
# 2. Similarity Search: It works by converting the user's question into the same vector space using 
#    the embeddings model. Then, it calculates the distance (e.g., L2 distance or Cosine Similarity) 
#    between the question's vector and the document chunk vectors.
# 3. Top K: The closest K vectors (e.g., K=4) are returned as the most semantically relevant text chunks.

def build_vector_store(chunks: List[Document], embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    """
    Creates and returns an in-memory FAISS vector database from a list of Document chunks.
    """
    # FAISS will compute the embedding for each chunk using the embeddings model 
    # and index them for rapid similarity searching.
    return FAISS.from_documents(chunks, embeddings)


def retrieve_relevant_documents(vector_store: FAISS, query: str, top_k: int = 4) -> List[Document]:
    """
    Queries the FAISS vector store to retrieve the top K most similar documents.
    """
    # similarity_search returns the Document objects that are closest in the embedding space
    return vector_store.similarity_search(query, k=top_k)


def retrieve_dynamic_documents(vector_store: FAISS, query: str, min_similarity: float = 0.35) -> List[Document]:
    """
    Retrieves a dynamic number of chunks based on semantic similarity scores
    to reduce context token usage and optimize RAG performance.
    """
    # Retrieve up to 3 documents with their distances
    docs_and_scores = vector_store.similarity_search_with_score(query, k=3)
    if not docs_and_scores:
        return []
        
    resolved_docs = []
    
    # Calculate similarity scores for each document
    for doc, dist in docs_and_scores:
        # Convert L2 distance to cosine similarity
        # Since all-MiniLM-L6-v2 outputs L2-normalized unit vectors,
        # cosine similarity = 1.0 - (dist ** 2) / 2.0
        similarity_score = 1.0 - (dist ** 2) / 2.0
        similarity_score = max(0.0, min(1.0, similarity_score))
        
        # Save score in document metadata
        doc.metadata["similarity_score"] = similarity_score
        resolved_docs.append(doc)
        
    if not resolved_docs:
        return []
        
    best_similarity_score = resolved_docs[0].metadata["similarity_score"]
    
    # If the best score is below the minimum threshold, we consider it irrelevant
    if best_similarity_score < min_similarity:
        return []
        
    # Dynamic selection of Top-K
    if best_similarity_score > 0.60:
        return resolved_docs[:1]
    elif best_similarity_score > 0.45:
        return resolved_docs[:2]
    else:
        return resolved_docs[:3]


def retrieve_documents_for_summary(vector_store: FAISS) -> List[Document]:
    """
    Retrieves representative and diverse chunks across the uploaded documents
    for global document summarization and overviews, ensuring document coverage
    within a strict context token budget.
    """
    all_docs = []
    try:
        if hasattr(vector_store, "docstore") and hasattr(vector_store.docstore, "_dict"):
            all_docs = list(vector_store.docstore._dict.values())
    except Exception:
        pass
        
    if not all_docs:
        return []
        
    # Group chunks by document source
    docs_by_source = {}
    for doc in all_docs:
        source = doc.metadata.get("source", "unknown")
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)
        
    num_docs = len(docs_by_source)
    if num_docs == 0:
        return []
        
    # Determine budget allocation per document
    if num_docs == 1:
        chunks_per_doc = 6
    elif num_docs <= 3:
        chunks_per_doc = 3
    elif num_docs <= 9:
        chunks_per_doc = 2
    else:
        chunks_per_doc = 1
        
    selected_docs = []
    
    # Select representative chunks for each document
    for source in sorted(docs_by_source.keys()):
        chunks = docs_by_source[source]
        # Sort chunks by page and chunk_index to ensure exact chronological order
        chunks.sort(key=lambda d: (d.metadata.get("page", 0), d.metadata.get("chunk_index", 0)))
        
        n = len(chunks)
        if n <= chunks_per_doc:
            selected_docs.extend(chunks)
        else:
            if chunks_per_doc == 6:
                # 6 evenly spaced chunks across beginning, early-middle, middle, late-middle, and end
                indices = [0, n // 5, (2 * n) // 5, (3 * n) // 5, (4 * n) // 5, n - 1]
                # Deduplicate indices in case n is small
                unique_indices = sorted(list(dict.fromkeys(indices)))
                for idx in unique_indices:
                    selected_docs.append(chunks[idx])
            elif chunks_per_doc == 3:
                # Beginning, middle, end
                selected_docs.append(chunks[0])
                selected_docs.append(chunks[n // 2])
                selected_docs.append(chunks[n - 1])
            elif chunks_per_doc == 2:
                # Beginning and end
                selected_docs.append(chunks[0])
                selected_docs.append(chunks[n - 1])
            else:
                # Middle (most representative single chunk)
                selected_docs.append(chunks[n // 2])
                
    # Strict total cap of 10 chunks
    return selected_docs[:10]
