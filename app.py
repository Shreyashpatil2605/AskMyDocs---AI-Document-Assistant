import os
import re
import hashlib
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import importlib
import src.pdf_processor
import src.embeddings
import src.vector_store
import src.rag_pipeline

importlib.reload(src.pdf_processor)
importlib.reload(src.embeddings)
importlib.reload(src.vector_store)
importlib.reload(src.rag_pipeline)

from src.pdf_processor import extract_documents_from_pdfs, split_documents
from src.embeddings import get_embeddings_model
from src.vector_store import build_vector_store, retrieve_relevant_documents, retrieve_dynamic_documents, retrieve_documents_for_summary
from src.rag_pipeline import generate_answer, needs_condensation, resolve_query_locally, is_global_document_query

# ---------------------------------------------------------
# Page Configuration & Styles
# ---------------------------------------------------------
st.set_page_config(
    page_title="AskMyDocs - AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

# Custom premium CSS styling to give a professional SaaS feel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* General styles */
    .stApp {
        background-color: #0d0f12;
        color: #f3f4f6;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 10px;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
        display: inline-block;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #11141a !important;
        border-right: 1px solid #1e2530;
    }
    
    .sidebar-header {
        font-weight: 700;
        font-size: 1.15rem;
        color: #60a5fa;
        margin-top: 15px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Custom File Cards in Sidebar */
    .file-card {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    .file-card:hover {
        border-color: #3b82f6;
        background-color: #1c212c;
        transform: translateY(-1px);
    }
    .file-name {
        font-weight: 600;
        font-size: 0.85rem;
        color: #c9d1d9;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 160px;
        display: inline-block;
    }
    .file-stats-text {
        font-size: 0.75rem;
        color: #8b949e;
        margin-top: 4px;
    }
    .status-badge {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 12px;
        text-transform: uppercase;
    }
    .status-badge.ready {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    .status-badge.pending {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }

    /* Hero section */
    .hero-container {
        text-align: center;
        padding: 50px 20px;
        margin-top: 20px;
        margin-bottom: 30px;
        background: radial-gradient(circle at top, rgba(59, 130, 246, 0.08) 0%, transparent 70%);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }
    .hero-title {
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #9ca3af;
        max-width: 650px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Feature Cards grid */
    .feature-card {
        background-color: #11141a;
        border: 1px solid #1e2530;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        height: 100%;
        transition: all 0.25s ease;
    }
    .feature-card:hover {
        border-color: #3b82f6;
        background-color: #161b26;
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.08);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 14px;
        display: inline-block;
    }
    .feature-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #f3f4f6;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.85rem;
        color: #9ca3af;
        line-height: 1.5;
    }

    /* Metric Cards Dashboard */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 20px;
        margin-top: 10px;
    }
    .metric-card {
        background-color: #11141a;
        border: 1px solid #1e2530;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        border-color: #2563eb;
        background-color: #141822;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #3b82f6;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* RAG Flowchart and status indicator */
    .rag-status-container {
        background: rgba(34, 197, 94, 0.03);
        border: 1px solid rgba(34, 197, 94, 0.15);
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 15px;
    }
    .rag-status-ready {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #4ade80;
    }
    .rag-flowchart {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        color: #9ca3af;
        font-weight: 500;
    }
    .flow-step {
        background-color: #161b22;
        border: 1px solid #21262d;
        padding: 4px 10px;
        border-radius: 6px;
        color: #c9d1d9;
    }
    .flow-arrow {
        color: #3b82f6;
        font-weight: bold;
    }

    /* Chat styling */
    .chat-bubble {
        padding: 16px 20px;
        border-radius: 14px;
        margin-bottom: 12px;
        line-height: 1.6;
        font-size: 0.95rem;
        white-space: pre-wrap;
        word-break: break-word;
        overflow-wrap: anywhere;
    }
    .chat-bubble-user {
        background-color: #1e293b;
        border: 1px solid #334155;
        color: #f1f5f9;
        margin-left: 15%;
        border-top-right-radius: 2px;
    }
    .chat-bubble-assistant {
        background-color: #111827;
        border: 1px solid #1f2937;
        color: #f9fafb;
        margin-right: 15%;
        border-top-left-radius: 2px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .chat-avatar {
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .chat-avatar-user {
        color: #60a5fa;
        margin-left: 15%;
    }
    .chat-avatar-assistant {
        color: #10b981;
    }

    /* Citation layout */
    .citations-section {
        margin-top: 15px;
        margin-bottom: 20px;
        padding: 12px 16px;
        background-color: #11141a;
        border: 1px solid #1e2530;
        border-radius: 12px;
    }
    .citations-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #9ca3af;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Streamlit expander override to look premium */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 8px !important;
        font-size: 0.8rem !important;
        color: #c9d1d9 !important;
        padding: 6px 12px !important;
    }
    .streamlit-expanderContent {
        background-color: #0d1117 !important;
        border: 1px solid #21262d !important;
        border-top: none !important;
        border-bottom-left-radius: 8px !important;
        border-bottom-right-radius: 8px !important;
        font-size: 0.8rem !important;
        font-style: italic;
        color: #8b949e !important;
        padding: 10px 14px !important;
    }

    /* Typing loading animation */
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 6px 0;
    }
    .typing-dot {
        width: 6px;
        height: 6px;
        background-color: #6b7280;
        border-radius: 50%;
        animation: typing-bounce 1.4s infinite ease-in-out both;
    }
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }

    @keyframes typing-bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1.0); }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False
if "processed_filenames" not in st.session_state:
    st.session_state.processed_filenames = []
if "file_stats" not in st.session_state:
    st.session_state.file_stats = {}
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = {}
if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = []
if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}

# Verify Google API Key is present
api_key = os.getenv("GOOGLE_API_KEY")
is_api_key_configured = bool(api_key and api_key.strip() and api_key != "your_google_gemini_api_key_here")

# ---------------------------------------------------------
# Sidebar Section (Upload & Settings)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">📂 Document Workspace</div>', unsafe_allow_html=True)
    
    # Display Google API Key Status
    if is_api_key_configured:
        st.success("Google Gemini API Key configured", icon="🔑")
    else:
        st.error("Google API key is missing. Please configure your .env file.", icon="⚠️")
        st.info("Get your API Key from [Google AI Studio](https://aistudio.google.com/).")
        
    # File Uploader (Accepts one or multiple PDF files)
    uploaded_files = st.file_uploader(
        "Upload PDF documents:",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload single or multiple PDF documents."
    )
    
    # Display list of uploaded files with premium card layouts
    if uploaded_files:
        st.markdown('<div style="margin-top: 10px; margin-bottom: 8px; font-weight: 600; font-size: 0.85rem; color: #60a5fa; text-transform: uppercase; letter-spacing: 0.5px;">Files In Workspace:</div>', unsafe_allow_html=True)
        for f in uploaded_files:
            is_ready = st.session_state.is_processed and f.name in st.session_state.processed_filenames
            stats = st.session_state.file_stats.get(f.name, {}) if is_ready else {}
            pages_count = stats.get("pages", 0)
            chunks_count = stats.get("chunks", 0)
            
            status_badge = '<span class="status-badge ready">Ready</span>' if is_ready else '<span class="status-badge pending">Pending</span>'
            stats_text = f'<div class="file-stats-text">📄 {pages_count} pages • 🧩 {chunks_count} chunks</div>' if is_ready else '<div class="file-stats-text">Click "Process Documents" to index</div>'
            
            st.markdown(f"""
            <div class="file-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 8px; overflow: hidden;">
                        <span style="font-size: 1.1rem;">📄</span>
                        <span class="file-name" title="{f.name}">{f.name}</span>
                    </div>
                    {status_badge}
                </div>
                {stats_text}
            </div>
            """, unsafe_allow_html=True)
    # Process Button
    process_clicked = st.button("Process Documents", type="primary", use_container_width=True)
    
    # Handle Process action
    if process_clicked:
        if not is_api_key_configured:
            st.sidebar.error("Google API key is missing. Please configure your .env file.")
        elif not uploaded_files:
            st.sidebar.warning("Please upload at least one PDF document.")
        else:
            # Sequence of checklist processing indicators
            with st.status("Initializing processing pipeline...", expanded=True) as status:
                try:
                    import hashlib
                    
                    def calculate_sha256(file_bytes) -> str:
                        return hashlib.sha256(file_bytes).hexdigest()
                    
                    # Calculate hashes for all currently uploaded files
                    current_uploads = []
                    for f in uploaded_files:
                        file_bytes = f.getvalue()
                        file_hash = calculate_sha256(file_bytes)
                        current_uploads.append((f, file_hash))
                        
                    new_files_to_process = []
                    chunks_to_add = []
                    new_processed_hashes = []
                    
                    for f, f_hash in current_uploads:
                        new_processed_hashes.append(f_hash)
                        if f_hash not in st.session_state.processed_docs:
                            new_files_to_process.append((f, f_hash))
                            
                    # Initialize / get embeddings model
                    embeddings_model = get_embeddings_model()
                    
                    # Process new files
                    if new_files_to_process:
                        status.update(label="⌛ Reading and chunking new PDF documents...", state="running")
                        new_pages = extract_documents_from_pdfs([f for f, _ in new_files_to_process])
                        
                        if not new_pages:
                            st.error("No readable text was found in the new PDFs.")
                            status.update(label="Failed to process documents", state="error")
                            raise ValueError("No readable text found")
                            
                        st.write(f"✓ Read and chunked {len(new_files_to_process)} new document(s)")
                        
                        # Split new pages
                        status.update(label="⌛ Splitting text into chunks...", state="running")
                        new_chunks = split_documents(new_pages)
                        
                        # Store in st.session_state.processed_docs
                        for f, f_hash in new_files_to_process:
                            file_chunks = [c for c in new_chunks if c.metadata["source"] == f.name]
                            file_pages = [p for p in new_pages if p.metadata["source"] == f.name]
                            
                            st.session_state.processed_docs[f_hash] = {
                                "filename": f.name,
                                "chunks": file_chunks,
                                "stats": {
                                    "pages": len(file_pages),
                                    "chunks": len(file_chunks)
                                }
                            }
                            chunks_to_add.extend(file_chunks)
                    else:
                        st.write("✓ All documents loaded from cache (skipped extraction and embedding)")
                        
                    # Build / Update FAISS
                    is_superset = set(st.session_state.processed_hashes).issubset(set(new_processed_hashes))
                    
                    if st.session_state.vector_store is not None and is_superset:
                        if chunks_to_add:
                            status.update(label="⌛ Adding new document vectors to FAISS...", state="running")
                            st.session_state.vector_store.add_documents(chunks_to_add)
                            st.write("✓ Added new document vectors to existing FAISS index")
                    else:
                        # Rebuild from scratch using cached chunks of all currently selected files
                        status.update(label="⌛ Building FAISS index...", state="running")
                        all_current_chunks = []
                        for _, f_hash in current_uploads:
                            if f_hash in st.session_state.processed_docs:
                                all_current_chunks.extend(st.session_state.processed_docs[f_hash]["chunks"])
                                
                        if all_current_chunks:
                            st.session_state.vector_store = build_vector_store(all_current_chunks, embeddings_model)
                            st.write(f"✓ Built FAISS index with {len(all_current_chunks)} total chunks")
                        else:
                            st.session_state.vector_store = None
                            
                    if set(st.session_state.processed_hashes) != set(new_processed_hashes):
                        st.session_state.answer_cache = {}
                    st.session_state.processed_hashes = new_processed_hashes
                    
                    # Construct file_stats and totals
                    file_stats = {}
                    total_pages = 0
                    total_chunks = 0
                    for f, f_hash in current_uploads:
                        if f_hash in st.session_state.processed_docs:
                            doc_data = st.session_state.processed_docs[f_hash]
                            file_stats[f.name] = doc_data["stats"]
                            total_pages += doc_data["stats"]["pages"]
                            total_chunks += doc_data["stats"]["chunks"]
                            
                    st.session_state.file_stats = file_stats
                    st.session_state.total_pages = total_pages
                    st.session_state.total_chunks = total_chunks
                    st.session_state.is_processed = True
                    st.session_state.processed_filenames = [f.name for f, _ in current_uploads]
                    
                    status.update(label="Documents Processed Successfully!", state="complete", expanded=False)
                    st.toast("Documents processed! Ask My Docs is ready.", icon="🚀")
                    st.rerun()
                    
                except Exception as e:
                    if str(e) != "No readable text found":
                        st.error(f"Error during document processing: {e}")
                        status.update(label="Processing Failed!", state="error")
    
    st.markdown('<div style="margin-top: 15px; margin-bottom: 15px; border-top: 1px solid #1e2530;"></div>', unsafe_allow_html=True)
    
    # Settings & Session Actions
    st.markdown('<div class="sidebar-header">🛠 Actions</div>', unsafe_allow_html=True)
    
    # New Chat: Clears history but preserves processed documents
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat history cleared. Kept active documents.", icon="🧹")
        st.rerun()
        
    # Reset All: Clears history and processed documents
    if st.button("Reset All", use_container_width=True, help="Clear documents and chat history"):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.is_processed = False
        st.session_state.processed_filenames = []
        st.session_state.file_stats = {}
        st.session_state.total_pages = 0
        st.session_state.total_chunks = 0
        st.session_state.processed_docs = {}
        st.session_state.processed_hashes = []
        st.session_state.answer_cache = {}
        st.toast("Workspace fully reset.", icon="🔄")
# ---------------------------------------------------------
def submit_question(question_text: str):
    if not is_api_key_configured:
        st.error("Google API key is missing. Please configure your .env file.")
        return
        
    if not st.session_state.is_processed:
        st.warning("Please upload and process your documents before asking questions.")
        return
        
    # Append user message and trigger rerun to execute response generation
    st.session_state.messages.append({"role": "user", "content": question_text})
    st.rerun()

# ---------------------------------------------------------
# Main Page UI Rendering
# ---------------------------------------------------------
st.markdown('<h1 class="main-title">AskMyDocs</h1>', unsafe_allow_html=True)

# Welcome Screen (Dashboard Onboarding)
if not st.session_state.is_processed:
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Chat with Your Documents Using AI</h1>
        <p class="hero-subtitle">
            Upload PDFs, retrieve relevant information using RAG, and get accurate AI-powered answers with source references.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render 3 Feature Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📄</span>
            <div class="feature-title">Upload & Analyze</div>
            <div class="feature-desc">Upload one or multiple PDF documents directly into the workspace.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🧠</span>
            <div class="feature-title">RAG-Powered Search</div>
            <div class="feature-desc">Semantic search retrieves the most relevant document context chunks.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">✨</span>
            <div class="feature-title">AI-Powered Answers</div>
            <div class="feature-desc">Gemini generates answers strictly grounded on your uploaded documents.</div>
        </div>
        """, unsafe_allow_html=True)

# Statistics & RAG Ready Status (Dashboard Statistics)
else:
    num_docs = len(st.session_state.processed_filenames)
    total_pages = st.session_state.total_pages
    total_chunks = st.session_state.total_chunks
    
    # 1. Metric cards
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-value">{num_docs}</div>
            <div class="metric-label">Files Uploaded</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_pages}</div>
            <div class="metric-label">Pages Processed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_chunks}</div>
            <div class="metric-label">Chunks Created</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color: #10b981; font-size: 1.5rem; padding-top: 5px;">🟢 Active</div>
            <div class="metric-label">Vector Search</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. RAG Flowchart status
    st.markdown("""
    <div class="rag-status-container">
        <div class="rag-status-ready">
            <span>🟢 RAG System Ready</span>
        </div>
        <div class="rag-flowchart">
            <span class="flow-step">Documents</span>
            <span class="flow-arrow">➔</span>
            <span class="flow-step">Chunks</span>
            <span class="flow-arrow">➔</span>
            <span class="flow-step">Embeddings</span>
            <span class="flow-arrow">➔</span>
            <span class="flow-step">FAISS</span>
            <span class="flow-arrow">➔</span>
            <span class="flow-step">Gemini</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Render Chat History
# ---------------------------------------------------------
if st.session_state.messages:
    st.markdown('<div style="margin-top: 20px; margin-bottom: 10px; font-weight: 600; font-size: 0.9rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px;">Chat Conversation:</div>', unsafe_allow_html=True)
    
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-avatar chat-avatar-user">👤 User</div>
        <div class="chat-bubble chat-bubble-user">{msg["content"]}</div>
        """, unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        st.markdown(f"""
        <div class="chat-avatar chat-avatar-assistant">🤖 Assistant</div>
        <div class="chat-bubble chat-bubble-assistant">{msg["content"]}</div>
        """, unsafe_allow_html=True)
        
        # Display Citation list cards if present
        sources = msg.get("sources", [])
        if sources:
            st.markdown("""
            <div class="citations-section">
                <div class="citations-title">🔍 Sources & Citations Used:</div>
            </div>
            """, unsafe_allow_html=True)
            
            displayed_sources = set()
            for idx, src_doc in enumerate(sources):
                doc_name = src_doc.metadata.get("source", "Unknown PDF")
                page_num = src_doc.metadata.get("page", "Unknown Page")
                
                # Deduplicate sources per page
                ref_key = f"{doc_name}_P{page_num}"
                if ref_key not in displayed_sources:
                    displayed_sources.add(ref_key)
                    snippet = src_doc.page_content.strip()
                    
                    # Create collapsible citation detail panel
                    with st.expander(f"📄 {doc_name} — Page {page_num}"):
                        st.markdown(f'"{snippet}"')

# ---------------------------------------------------------
# Response Generation Engine (with Typing Animation)
# ---------------------------------------------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    
    # Display the typing indicator block
    with st.empty():
        st.markdown("""
        <div class="chat-avatar chat-avatar-assistant">🤖 Assistant</div>
        <div class="chat-bubble chat-bubble-assistant">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Exclude the current user query from the history check since it was just appended
            chat_history_excluding_current = st.session_state.messages[:-1]
            
            # Check if query needs condensation
            if needs_condensation(user_query, chat_history_excluding_current):
                retrieval_query = resolve_query_locally(user_query, chat_history_excluding_current)
            else:
                retrieval_query = user_query
                
            # Helper to normalize query for cache stability
            def normalize_for_cache(q: str) -> str:
                q_clean = q.lower().strip()
                q_clean = re.sub(r"[?.,!]", "", q_clean)
                return " ".join(q_clean.split())
                
            # Compute stable document-aware cache key
            sorted_hashes = sorted(st.session_state.processed_hashes)
            cache_payload = (
                "|".join(sorted_hashes)
                + "|"
                + normalize_for_cache(user_query)
                + "|"
                + normalize_for_cache(retrieval_query)
            )
            cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
            
            # Look up answer cache
            if cache_key in st.session_state.answer_cache:
                cached_res = st.session_state.answer_cache[cache_key]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": cached_res["answer"],
                    "sources": cached_res["sources"]
                })
            else:
                # 1. Select the retrieval strategy based on query intent
                if is_global_document_query(user_query):
                    docs = retrieve_documents_for_summary(st.session_state.vector_store)
                else:
                    docs = retrieve_dynamic_documents(st.session_state.vector_store, retrieval_query)
                
                if not docs:
                    # No relevant context -> Return clean message without invoking Gemini
                    answer = "I could not find any relevant information in the uploaded documents to answer your question."
                    result = {
                        "answer": answer,
                        "sources": []
                    }
                else:
                    # 2. Run RAG Pipeline using LLM with conversation history
                    result = generate_answer(user_query, docs, chat_history=chat_history_excluding_current)
                    
                # Cache response
                st.session_state.answer_cache[cache_key] = {
                    "answer": result["answer"],
                    "sources": result["sources"]
                }
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Sorry, I encountered an issue while generating an answer. Please verify your Google API key or try again.",
                "sources": []
            })
            
        st.rerun()

# ---------------------------------------------------------
# Suggested Questions Panel
# ---------------------------------------------------------
if st.session_state.is_processed:
    st.write("---")
    st.write("💡 **Suggested Questions:**")
    suggestions = [
        "Summarize the uploaded documents.",
        "What are the main topics?",
        "What are the important points?",
        "Explain the most important concept simply."
    ]
    
    # Create 4 columns for buttons
    cols = st.columns(len(suggestions))
    for idx, sug in enumerate(suggestions):
        if cols[idx].button(sug, key=f"sug_btn_{idx}", use_container_width=True):
            submit_question(sug)

# ---------------------------------------------------------
# Chat Input Bar & Setup Help
# ---------------------------------------------------------
if not st.session_state.is_processed:
    st.info("💡 Please upload and process your PDF documents in the sidebar to start asking questions.")
    
# Always render input box at the bottom, handling state blockages gracefully
prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    if not st.session_state.is_processed:
        st.warning("Please upload and process your documents before asking questions.")
    else:
        submit_question(prompt)
