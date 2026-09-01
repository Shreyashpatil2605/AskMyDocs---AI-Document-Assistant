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

---

## 🎓 Technical Interview Guide

Here is how you can explain this project in a placement interview:

### 1. What is RAG?
"RAG stands for Retrieval-Augmented Generation. Instead of relying solely on what an LLM learned during training (which has a knowledge cutoff and can hallucinate), RAG retrieves relevant facts from a local source of truth (like uploaded PDFs) and feeds them directly to the LLM alongside the user's question. This makes answers factual, grounded, and specific."

### 2. Why do we chunk documents?
"We chunk documents because LLMs have context window constraints and processing large documents is expensive and slow. Chunking allows us to break down pages into smaller, overlaps-inclusive snippets (we used 1000 characters with 200 overlap). This ensures that only the relevant sections are retrieved, saving cost and improving answer precision."

### 3. What are Embeddings?
"Embeddings are high-dimensional vector representations of text. An embedding model converts words or sentences into list of floats, capturing their semantic meaning. Concepts with similar meanings are close together in the vector space. We used Google's `models/gemini-embedding-001`."

### 4. What is FAISS and why is it used?
"FAISS stands for Facebook AI Similarity Search. It is a highly optimized library for finding similar vectors. Instead of doing a linear search (O(N)) comparing the query embedding with all document chunks, FAISS indexes vectors for ultra-fast distance calculations (like Cosine Similarity or L2 distance)."

### 5. How does Gemini generate grounded answers?
"We retrieve the top 4 chunks, merge them into a single string, and inject them into a Prompt Template. The prompt tells Gemini to answer *only* from the text and output a specific fallback string if the answer is missing. We set `temperature=0.0` to force deterministic behavior and prevent creativity/hallucinations."
