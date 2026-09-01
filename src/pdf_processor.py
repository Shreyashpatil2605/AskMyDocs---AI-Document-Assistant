import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Interview Tip: Why do we chunk documents?
# 1. Context Window Limits: LLMs have a maximum limit of tokens they can process in a single request.
# 2. Precision: Retrieving a 100-page document for a single question adds noise. Chunking helps us find
#    and retrieve only the specific paragraphs (e.g., 1000 characters) relevant to the query.
# 3. Cost & Speed: Smaller, relevant contexts use fewer API tokens and generate faster responses.
# 4. Chunk Overlap: The `chunk_overlap` (e.g., 200) ensures that semantic context split at boundaries is 
#    preserved, avoiding cut-off sentences or missing context between chunks.

def extract_documents_from_pdfs(uploaded_files, temp_dir: str = "data") -> List[Document]:
    """
    Extracts pages from uploaded PDF files, cleans their metadata, 
    and handles empty pages safely.
    """
    # Ensure temporary directory exists
    os.makedirs(temp_dir, exist_ok=True)
    
    all_pages = []
    
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily to the local disk
        # This is required because PyPDFLoader works with physical file paths.
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Initialize LangChain's PyPDFLoader
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            
            for idx, page in enumerate(pages):
                # Clean up metadata. We override the temporary filepath source
                # with the actual filename for user-friendly display in references.
                page.metadata["source"] = uploaded_file.name
                page.metadata["page"] = page.metadata.get("page", idx) + 1  # 1-indexed page
                
                # Strip text and verify if it's empty
                # Some PDFs contain scanned image pages with no embedded text (safely ignored).
                if page.page_content.strip():
                    all_pages.append(page)
                    
        finally:
            # Clean up the temporary file immediately to keep environment clean
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    return all_pages


def split_documents(documents: List[Document], chunk_size: int = 1500, chunk_overlap: int = 100) -> List[Document]:
    """
    Splits larger documents or pages into smaller chunks.
    Uses RecursiveCharacterTextSplitter which splits text by double newlines, single newlines,
    spaces, and characters recursively to maintain paragraph/sentence integrity.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Track index of chunks for each source document
    source_counts = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        if source not in source_counts:
            source_counts[source] = 0
        chunk.metadata["chunk_index"] = source_counts[source]
        source_counts[source] += 1
        
    return chunks
