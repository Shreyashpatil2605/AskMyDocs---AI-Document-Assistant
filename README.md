# 📄 AskMyDocs - AI Document Assistant

> **Chat with your PDF documents using Retrieval-Augmented Generation
> (RAG).**

AskMyDocs is an AI-powered document assistant that lets users upload one
or multiple PDF documents, process them into semantic chunks, retrieve
relevant information using embeddings and FAISS, and ask questions in
natural language. Google Gemini generates answers grounded in the
retrieved document context, with source citations for transparency.

## 🚀 Live Demo
<img width="1917" height="840" alt="image" src="https://github.com/user-attachments/assets/ec81bf0a-b2e6-4de8-a33d-9723d5a82177" />




**[Open
AskMyDocs](https://askmydocs---ai-document-assistant-m6to5p3besakvzt7n6nwc8.streamlit.app/)**

## ✨ Features

-   📄 **Multiple PDF Uploads** --- Upload one or several PDF documents
    in a single workspace.
-   🧩 **Document Chunking** --- Splits extracted document text into
    smaller, retrieval-friendly chunks.
-   🧠 **Semantic Embeddings** --- Converts document chunks into vector
    representations for semantic search.
-   ⚡ **FAISS Vector Search** --- Retrieves relevant document chunks
    efficiently.
-   🤖 **Gemini-Powered Answers** --- Generates answers using retrieved
    document context.
-   💬 **Conversational Q&A** --- Supports follow-up questions using
    chat history.
-   🔄 **Query Condensation** --- Resolves context-dependent follow-up
    questions before retrieval.
-   📚 **Global Document Queries** --- Uses a document-level retrieval
    strategy for questions such as summaries and main topics.
-   🔍 **Source Citations** --- Displays the PDF filename, page number,
    and source text used for an answer.
-   💾 **Session Caching** --- Avoids unnecessary re-processing and
    repeated answer generation within a session.
-   📊 **Workspace Dashboard** --- Shows uploaded files, processed
    pages, generated chunks, and vector-search status.
-   🎨 **Responsive UI** --- Custom Streamlit styling for a clean, dark
    SaaS-style interface.

## 🧠 How It Works

AskMyDocs follows a Retrieval-Augmented Generation pipeline:

``` text
             PDF Upload
                  │
                  ▼
          Text Extraction
                  │
                  ▼
          Document Chunking
                  │
                  ▼
        Generate Embeddings
                  │
                  ▼
           FAISS Vector Store
                  │
          ┌───────┴────────┐
          │                │
     User Question    Chat History
          │                │
          └───────┬────────┘
                  ▼
          Query Processing
                  │
                  ▼
       Semantic Retrieval
                  │
                  ▼
        Relevant PDF Chunks
                  │
                  ▼
            Google Gemini
                  │
                  ▼
       Grounded AI Response
                  │
                  ▼
        Sources & Citations
```

### RAG Pipeline

1.  The user uploads one or more PDFs.
2.  Text and page information are extracted from the documents.
3.  Documents are split into smaller chunks.
4.  Embeddings are generated for the chunks.
5.  The embeddings are stored in a FAISS vector index.
6.  The user asks a question.
7.  The application determines the appropriate retrieval strategy.
8.  Relevant chunks are retrieved from FAISS.
9.  The retrieved context and conversation history are provided to
    Gemini.
10. Gemini generates a grounded response.
11. The application displays the answer together with source citations.

## 🛠️ Tech Stack

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  **Python**                          Core application and RAG logic

  **Streamlit**                       Web UI and application framework

  **Google Gemini**                   Large Language Model for answer
                                      generation

  **Embeddings**                      Semantic representation of document
                                      chunks

  **FAISS**                           Vector similarity search

  **PDF Processing**                  Extracting text and page-level
                                      document information

  **python-dotenv**                   Local environment variable
                                      management

  **Git & GitHub**                    Version control and source hosting

  **Streamlit Community Cloud**       Deployment
  -----------------------------------------------------------------------

## 📁 Project Structure

``` text
AskMyDocs---AI-Document-Assistant/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── sample.pdf                # Sample document for testing
│
├── src/
│   ├── pdf_processor.py      # PDF extraction and document chunking
│   ├── embeddings.py         # Embedding model setup
│   ├── vector_store.py       # FAISS indexing and retrieval
│   └── rag_pipeline.py       # Query processing and Gemini generation
│
├── data/                     # Document/data resources
├── scratch/                  # Temporary working files
├── .gitignore                # Git ignore rules
└── .env                      # Local secrets — NOT committed
```

## ⚙️ Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/Shreyashpatil2605/AskMyDocs---AI-Document-Assistant.git
cd AskMyDocs---AI-Document-Assistant
```

### 2. Create a virtual environment

Windows:

``` powershell
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

``` env
GOOGLE_API_KEY=your_google_gemini_api_key
```

**Never commit `.env` to GitHub.**

The project includes `.env` in `.gitignore`.

### 5. Run the application

``` bash
streamlit run app.py
```

The application will normally be available at:

``` text
http://localhost:8501
```

## 🔐 Environment Variables

The application requires:

  Variable           Description
  ------------------ ----------------------------------------------
  `GOOGLE_API_KEY`   Google Gemini API key used for AI generation

For the deployed Streamlit application, configure the key through
**Streamlit Secrets** instead of committing it to the repository.

Example:

``` toml
GOOGLE_API_KEY = "your_google_gemini_api_key"
```

## 💡 Example Questions

After uploading and processing a PDF, you can ask:

``` text
Summarize the uploaded documents.
```

``` text
What are the main topics discussed?
```

``` text
What are the important points?
```

``` text
Explain the most important concept simply.
```

You can also ask follow-up questions that depend on previous
conversation context.

## 🔎 Source Citations

AskMyDocs is designed to make RAG responses more transparent.

For each generated answer, the application can display:

-   📄 Source PDF filename
-   📑 Page number
-   🔍 Retrieved source text

This helps the user verify where the answer came from instead of relying
on an unsupported response.

## 🧠 Key GenAI Concepts Demonstrated

This project demonstrates practical implementation of:

-   Large Language Models (LLMs)
-   Retrieval-Augmented Generation (RAG)
-   Text chunking
-   Embeddings
-   Vector representations
-   Vector similarity search
-   FAISS
-   Semantic retrieval
-   Query condensation
-   Conversation-aware retrieval
-   Context management
-   Prompt-based answer generation
-   Grounded question answering
-   Source attribution

## 🧪 Caching & Efficiency

AskMyDocs uses session-level caching to reduce unnecessary computation.

### Document caching

Uploaded documents are identified using SHA-256 hashes. Previously
processed documents can be reused instead of extracting and embedding
the same document again during the session.

### Answer caching

A document-aware cache key is generated from:

-   Processed document hashes
-   Original user question
-   Retrieval query

This allows repeated questions against the same document set to reuse
previously generated answers during the session.

## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment flow:

``` text
GitHub Repository
       │
       ▼
Streamlit Community Cloud
       │
       ├── app.py
       ├── src/
       ├── requirements.txt
       └── Streamlit Secrets
              │
              ▼
       Google Gemini API
```

### Deployment configuration

-   **Main file:** `app.py`
-   **Branch:** `main`
-   **Python:** 3.11 recommended for dependency compatibility
-   **Secret:** `GOOGLE_API_KEY`

## ⚠️ Current Scope

AskMyDocs is designed as a portfolio/demo RAG application.

The current architecture keeps processed documents, FAISS state, chat
history, and caches in Streamlit session state. This means the processed
workspace is session-based rather than a permanent multi-user document
database.

For a larger production system, the architecture could be extended with
persistent storage and a production vector database.

## 🔮 Future Improvements

-   🔐 User authentication and private document workspaces
-   ☁️ Persistent document storage
-   🗄️ Persistent vector database
-   📈 Usage and retrieval analytics
-   🧾 Improved citation highlighting
-   📚 Support for additional document formats
-   ⚡ Background document processing
-   🧠 Hybrid keyword + semantic retrieval
-   🔄 Streaming LLM responses
-   🧑‍🤝‍🧑 Multi-user document isolation
-   🗑️ Persistent document management and deletion

## 👨‍💻 Author

**Shreyash Patil**

Computer Science & Engineering

### Project

**AskMyDocs --- AI Document Assistant**

Built to demonstrate a practical end-to-end **GenAI + RAG** application
using document ingestion, embeddings, vector search, and LLM-based
grounded generation.

------------------------------------------------------------------------

⭐ If you find this project useful, consider giving the repository a
star!
