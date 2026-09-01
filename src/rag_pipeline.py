import os
import re
import time
from typing import Dict, Any, List
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

def invoke_llm_with_retry(llm: ChatGoogleGenerativeAI, prompt: str) -> Any:
    """
    Helper function to invoke the Gemini LLM with automatic 429 rate limit retry logic.
    Supports parsing dynamic delays from the Gemini API response message or falling back
    to exponential backoff, complete with Streamlit warning overlays.
    """
    max_retries = 3
    base_delay = 5.0
    
    attempt = 0
    while attempt < max_retries:
        try:
            return llm.invoke(prompt)
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = (
                "429" in err_msg or 
                "resource_exhausted" in err_msg or 
                "rate limit" in err_msg or 
                "quota" in err_msg or
                (hasattr(e, "code") and e.code == 429) or
                (hasattr(e, "status_code") and e.status_code == 429)
            )
            
            if is_rate_limit:
                attempt += 1
                if attempt >= max_retries:
                    raise RuntimeError("Gemini API rate limit reached. Please wait a minute and try again.") from e
                
                # Dynamic delay extraction from error message
                delay = 0.0
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
                if match:
                    delay = float(match.group(1))
                else:
                    match = re.search(r"retrydelay'?: '?:?(\d+)s", err_msg)
                    if match:
                        delay = float(match.group(1))
                
                # Fallback to exponential backoff
                if delay <= 0.0:
                    delay = base_delay * (2 ** attempt)
                
                # Limit max delay to 60s
                delay = min(delay, 60.0)
                
                msg_text = (
                    f"⏳ Gemini free-tier limit reached. Your question is queued for retry. "
                    f"Waiting {delay:.1f} seconds before retrying (Attempt {attempt}/{max_retries})..."
                )
                try:
                    st.warning(msg_text)
                except Exception:
                    print(msg_text)
                    
                time.sleep(delay)
            else:
                raise e


def needs_condensation(query: str, chat_history: List[Dict[str, str]]) -> bool:
    """
    Lightweight local heuristic to determine if the query is ambiguous or depends
    on the previous conversation context.
    Checks for relative terms, pronouns, or references in the user's question.
    Does NOT call Gemini API.
    """
    # If there is no chat history, condensation is never needed
    if not chat_history:
        return False
        
    query_lower = query.lower().strip()
    
    # Check for references or pronouns (using strict word boundaries)
    ref_words = [
        r"\bit\b", r"\bthis\b", r"\bthat\b", r"\bthey\b", r"\bthose\b",
        r"\bthese\b", r"\bthem\b", r"\btheir\b", r"\bits\b",
        r"\bhim\b", r"\bher\b", r"\bhe\b", r"\bshe\b",
        r"\bagain\b", r"\babove\b", r"\bprevious\b", r"\bformer\b", r"\blatter\b",
        r"\bdo so\b", r"\bsuch\b", r"\bsame\b",
        r"\bsecond one\b", r"\bfirst one\b", r"\blast one\b"
    ]
    
    for word in ref_words:
        if re.search(word, query_lower):
            return True
            
    return False


def format_chat_history(chat_history: List[Dict[str, str]], max_turns: int = 2) -> str:
    """
    Formats the last N complete conversation turns into a string.
    Each turn has a User message and an Assistant response.
    """
    if not chat_history:
        return "No previous conversation."
        
    turns = []
    i = len(chat_history) - 1
    while i >= 0 and len(turns) < max_turns:
        msg = chat_history[i]
        if msg["role"] == "assistant":
            if i > 0 and chat_history[i - 1]["role"] == "user":
                user_msg = chat_history[i - 1]["content"]
                asst_msg = msg["content"]
                turns.append((user_msg, asst_msg))
                i -= 2
            else:
                turns.append(("", msg["content"]))
                i -= 1
        else:
            i -= 1
            
    turns.reverse()
    
    formatted = ""
    for u, a in turns:
        if u:
            formatted += f"User: {u}\n"
        formatted += f"Assistant: {a}\n\n"
        
    return formatted.strip()


def resolve_query_locally(query: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Resolves context-dependent follow-up queries locally by appending the previous
    user query as context.
    Does NOT call any external API.
    """
    if not chat_history:
        return query
        
    # Find the most recent user message
    last_user_msg = None
    for msg in reversed(chat_history):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "").strip()
            break
            
    if last_user_msg:
        p_query = last_user_msg.rstrip("?.!")
        c_query = query.strip()
        return f"{p_query}? {c_query}"
        
    return query


def generate_answer(
    query: str, 
    retrieved_docs: List[Document],
    chat_history: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Combines retrieved chunks and relevant conversation history (max 2 turns),
    formats the RAG prompt, sends it to Google Gemini, and returns the grounded response.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key is missing. Please configure your .env file or input it in the sidebar.")

    # 1. Format the retrieved context blocks
    context = ""
    for idx, doc in enumerate(retrieved_docs):
        src_name = doc.metadata.get("source", "Unknown PDF")
        page_num = doc.metadata.get("page", "Unknown Page")
        context += f"--- Document Chunk {idx + 1} (Source: {src_name}, Page: {page_num}) ---\n"
        context += f"{doc.page_content}\n\n"

    # 2. Format chat history (last 2 turns)
    history_text = format_chat_history(chat_history, max_turns=2)

    # 3. Format RAG prompt
    from src.prompts import RAG_PROMPT_TEMPLATE
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=RAG_PROMPT_TEMPLATE
    )
    formatted_prompt = prompt_template.format(
        chat_history=history_text,
        context=context,
        question=query
    )

    # 4. Initialize LLM and invoke
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0.0,
        max_output_tokens=4096
    )

    response = invoke_llm_with_retry(llm, formatted_prompt)
    answer = response.content
    
    if isinstance(answer, list):
        text_parts = []
        for part in answer:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        answer = "".join(text_parts)
    elif not isinstance(answer, str):
        answer = str(answer)

    return {
        "answer": answer,
        "sources": retrieved_docs
    }


def is_global_document_query(query: str) -> bool:
    """
    Locally detects if the query is asking for a broad/global document overview or summary,
    without making any external LLM/API calls.
    """
    q = query.lower().strip()
    
    # List of key phrases that indicate global intent
    global_phrases = [
        "summarize",
        "summary",
        "overview",
        "main topics",
        "main points",
        "important points",
        "key points",
        "explain the document",
        "what is this document about",
        "what are the documents about",
        "describe the document",
        "give me an overview",
        "analyze the document",
        "what is this about",
        "what are these about"
    ]
    
    for phrase in global_phrases:
        if phrase in q:
            return True
            
    # Also check regex patterns for global document intents
    patterns = [
        r"\bsummariz(e|ation)\b",
        r"\bsummar(y|ies)\b",
        r"\boverview\b",
        r"\bkey\s+points\b",
        r"\bmain\s+topics\b",
        r"\bmain\s+points\b",
        r"\bimportant\s+points\b",
        r"\bwhat\s+is\s+this\s+about\b",
        r"\bwhat\s+are\s+these\s+about\b",
        r"\bexplain\s+the\s+document\b",
        r"\bdescribe\s+the\s+document\b",
        r"\banalyze\s+the\s+document\b"
    ]
    
    for pattern in patterns:
        if re.search(pattern, q):
            return True
            
    return False
