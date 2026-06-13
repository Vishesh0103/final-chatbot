import streamlit as st
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from serpapi import GoogleSearch
import tempfile
import os
import regex as re

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ── Load API keys from Streamlit secrets ───────────────────────────────────────
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SERPAPI_KEY    = st.secrets["SERPAPI_KEY"]

# ── Configure Gemini ───────────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=(
        "You are a helpful, friendly, and smart AI assistant. "
        "Answer clearly and concisely. If you are given document context, "
        "use it to answer the question accurately."
    ),
)

# ── Session state defaults ─────────────────────────────────────────────────────
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "vector_store"  not in st.session_state: st.session_state.vector_store  = None
if "doc_names"     not in st.session_state: st.session_state.doc_names     = []

SIMILARITY_THRESHOLD = 0.60   # lower = stricter match required for RAG

# ── Helper: build / update vector store ───────────────────────────────────────
def add_documents_to_store(uploaded_files):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY,
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:                           # .txt and other plain text
                loader = TextLoader(tmp_path, encoding="utf-8")

            docs   = loader.load()
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

            if uploaded_file.name not in st.session_state.doc_names:
                st.session_state.doc_names.append(uploaded_file.name)
        finally:
            os.unlink(tmp_path)

    if all_chunks:
        if st.session_state.vector_store is None:
            st.session_state.vector_store = FAISS.from_documents(all_chunks, embeddings)
        else:
            st.session_state.vector_store.add_documents(all_chunks)
        return True
    return False


# ── Helper: search the web via SerpAPI ────────────────────────────────────────
def web_search(query: str) -> str:
    params = {"q": query, "api_key": SERPAPI_KEY, "num": 5}
    results = GoogleSearch(params).get_dict()

    snippets = []
    for r in results.get("organic_results", [])[:5]:
        title   = r.get("title", "")
        snippet = r.get("snippet", "")
        link    = r.get("link", "")
        if snippet:
            snippets.append(f"**{title}**\n{snippet}\n({link})")

    return "\n\n".join(snippets) if snippets else "No relevant web results found."


# ── Helper: decide source and generate answer ──────────────────────────────────
def get_answer(user_query: str) -> tuple[str, str]:
    """Returns (answer, source_label)."""
    # 1️⃣  Try RAG first
    if st.session_state.vector_store is not None:
        results = st.session_state.vector_store.similarity_search_with_score(user_query, k=3)
        if results:
            best_doc, score = results[0]
            if score < SIMILARITY_THRESHOLD:
                context = "\n\n".join([d.page_content for d, _ in results])
                prompt  = (
                    f"Use ONLY the context below to answer the question.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {user_query}"
                )
                response = model.generate_content(prompt)
                return response.text, "📄 Document (RAG)"

    # 2️⃣  Try Gemini's own knowledge
    response  = model.generate_content(user_query)
    relevance = model.generate_content(
        f"Question: '{user_query}'\nResponse: '{response.text}'\n"
        "Classify as 'Relevant' or 'Not Relevant'. Reply with ONLY those words."
    )
    if "relevant" in relevance.text.strip().lower() and "not" not in relevance.text.strip().lower():
        return response.text, "🧠 Gemini AI"

    # 3️⃣  Fallback: web search
    web_context = web_search(user_query)
    prompt = (
        f"Use the following web search results to answer the question.\n\n"
        f"Search Results:\n{web_context}\n\n"
        f"Question: {user_query}"
    )
    final = model.generate_content(prompt)
    return final.text, "🌐 Web Search"


# ── Build conversation history string for context ──────────────────────────────
def build_history_context() -> str:
    if not st.session_state.chat_history:
        return ""
    lines = []
    for msg in st.session_state.chat_history[-6:]:   # last 3 turns
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════════════════════════════════════

st.title("🤖 Smart Chatbot")
st.caption("Powered by Gemini · RAG · Web Search")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Documents")
    st.write("Upload **PDF** or **TXT** files to chat with your own knowledge base.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("➕ Add to Knowledge Base", use_container_width=True):
            with st.spinner("Processing documents…"):
                success = add_documents_to_store(uploaded_files)
            if success:
                st.success(f"✅ Added {len(uploaded_files)} file(s)!")
            else:
                st.error("Could not process the files. Please check the format.")

    if st.session_state.doc_names:
        st.divider()
        st.subheader("📚 Loaded Documents")
        for name in st.session_state.doc_names:
            st.markdown(f"- `{name}`")

    st.divider()
    st.subheader("⚙️ How it works")
    st.markdown(
        """
        1. **📄 RAG** – searches your documents first  
        2. **🧠 Gemini AI** – uses its own knowledge  
        3. **🌐 Web Search** – fetches live results as fallback
        """
    )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Chat display ───────────────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "source" in msg:
                st.caption(f"Source: {msg['source']}")

# ── Chat input ─────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask me anything…"):
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            # Prepend short memory context to query
            history_ctx = build_history_context()
            query_with_ctx = (
                f"Previous conversation:\n{history_ctx}\n\nNew question: {user_input}"
                if history_ctx else user_input
            )
            answer, source = get_answer(query_with_ctx)

        st.markdown(answer)
        st.caption(f"Source: {source}")

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "source": source}
    )
