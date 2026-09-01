import os
from langchain_community.embeddings import HuggingFaceEmbeddings

# Interview Tip: What are Text Embeddings?
# 1. Definition: Embeddings are dense vector representations of text in a high-dimensional space
#    (e.g., 384 dimensions for all-MiniLM-L6-v2).
# 2. Semantic Meaning: Instead of matching exact keywords (like traditional search), embeddings capture 
#    the meaning of sentences. Words/sentences with similar meanings (e.g., "king" and "queen", or 
#    "AI assistant" and "chatbot") are placed close together in this vector space.
# 3. Local Model: We use the popular "sentence-transformers/all-MiniLM-L6-v2" model which runs locally.
#    This completely eliminates Google Gemini embedding API requests during document processing, 
#    preventing quota limits while delivering fast similarity search.

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the local SentenceTransformers embedding model.
    Runs entirely locally on the host CPU or GPU.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
