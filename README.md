# AskMyDocs – AI Document Assistant

AskMyDocs is a complete, production-quality, yet beginner-friendly Generative AI application built on the **Retrieval-Augmented Generation (RAG)** architecture. It allows users to upload single or multiple PDF documents, automatically parse and index their contents, and ask natural language questions. The assistant answers based *strictly* on the document context, citing references to source files and page numbers.

---

## 🚀 Features

1. **Modern Chat UI**: Clean, professional Streamlit-based interface designed with custom CSS styled after popular AI chat applications.
2. **Multiple PDF Upload**: Support for uploading and processing single or multiple PDF documents simultaneously.
3. **Pipeline Progress Indicators**: Visual step-by-step progress tracking (reading, extracting, chunking, embedding, indexing, ready).
4. **Smart Metadata Mapping**: Tracks source file names and actual page numbers (1-indexed) across all uploaded files.
5. **Robust RAG Core**: Utilizes Recursive Character Text Splitting, local FAISS vector store indexes, and Google's Gemini API (`gemini-3.6-flash`).
6. **Zero Hallucination Prompts**: Enforces strict prompt boundaries directing the LLM to only answer from the provided documents or explicitly output: *"I could not find this information in the uploaded documents."*
7. **Source Citations & Explanations**: Every response lists the sources and pages utilized along with short text snippets.
8. **Interactive Suggested Questions**: Dynamic buttons appearing after document processing for quick queries.
9. **Environment Configuration**: Uses `.env` for secrets isolation, avoiding hardcoded API credentials.

---

## 🛠 Technologies Used

- **Python 3.8+**
- **Streamlit**: Web UI framework.
- **LangChain**: LLM orchestration and workflow framing.
- **Google Gemini API**: Via `langchain-google-genai` for embedding (`models/gemini-embedding-001`) and answers (`gemini-3.6-flash`).
- **FAISS (Facebook AI Similarity Search)**: Efficient vector similarity lookup database.
- **PyPDF**: PDF loading and page parsing.
- **python-dotenv**: Configuration management.

---

## 📐 RAG Architecture

```
                                 [ User Question ]
                                         │
                                         ▼
[ Upload PDFs ]                  [ Generate Query Vector ]
       │                                 │
       ▼                                 ▼
[ Extract Pages ]              ┌───────────────────┐
       │                       │   FAISS Vector    │
       ▼                       │     Database      │
[ Chunk Text ] ──────────────> │                   │
       │                       │(Similarity Search)│
       ▼                       └─────────┬─────────┘
[ Create Embeddings ]                    │
                                         ▼
                               [ Retrieve Top K=4 ]
                                         │
                                         ▼
                               [ Augment Prompt ]
                                         │
                                         ▼
                               [ Google Gemini API ]
                                         │
                                         ▼
                                [ Factual Answer ]
```

---

## 💻 Installation & Setup

Follow these steps to run AskMyDocs locally on your machine:

### 1. Clone the repository
```bash
git clone <repository-url>
cd AskMyDocs
```

### 2. Create and Activate a Virtual Environment
**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your Gemini API Key:
   ```env
   GOOGLE_API_KEY=your_actual_google_gemini_api_key
   ```
   *Note: You can get your key for free or pay-as-you-go from [Google AI Studio](https://aistudio.google.com/).*

### 5. Run the Application
```bash
streamlit run app.py
```

