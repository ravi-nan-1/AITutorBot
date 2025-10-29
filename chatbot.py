import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from typing import Optional, List
from urllib.parse import urljoin, urlparse
import urllib3
import os
import time
import numpy as np

# For offline models
from sentence_transformers import SentenceTransformer
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Offline Model Classes ---

class LocalEmbeddings(Embeddings):
    """Custom embeddings class using sentence-transformers for offline use"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()

class LocalLLM(LLM):
    """Custom LLM class using Ollama for offline use"""
    
    model_name: str
    base_url: str = "http://localhost:11434"
    
    def __init__(self, model_name: str = "llama2"):
        super().__init__()
        self.model_name = model_name
        # Initialize Ollama
        self.ollama = Ollama(model=model_name, base_url=self.base_url)
    
    @property
    def _llm_type(self) -> str:
        return "local_ollama"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = self.ollama.invoke(prompt)
            return response
        except Exception as e:
            st.error(f"Error calling Ollama: {e}. Make sure Ollama is running locally.")
            return "Error: Could not get response from local model."

# --- Helper Functions ---

def extract_text_from_url(url: str) -> str:
    try:
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/114.0.0.0 Safari/537.36")
        }
        res = requests.get(url, headers=headers, timeout=10, verify=False)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        cleaned = [line for line in lines if line]
        return "\n".join(cleaned)
    except Exception as e:
        st.error(f"Error extracting from {url}: {e}")
        return ""

def check_ollama_status():
    """Check if Ollama is running locally"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            return True, response.json().get('models', [])
        return False, []
    except:
        return False, []

def get_available_embedding_models():
    """Get list of available sentence-transformer models"""
    # Common lightweight models that work well offline
    return [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
        "paraphrase-multilingual-MiniLM-L12-v2"
    ]

# --- Streamlit UI ---

st.title("Offline Contextual Q&A from URLs")

# Check Ollama status
ollama_running, ollama_models = check_ollama_status()

if not ollama_running:
    st.error("⚠️ Ollama is not running! Please install and start Ollama:")
    st.code("# Install Ollama from https://ollama.ai\n# Then run:\nollama serve")
    st.info("After starting Ollama, pull a model (e.g., `ollama pull llama2`) and refresh this page.")
    st.stop()

# Model selection
st.subheader("Select Models")

# Chat model selection (from Ollama)
if ollama_models:
    model_names = [model['name'] for model in ollama_models]
    selected_chat_model = st.selectbox("Select Chat Model (Ollama)", model_names, index=0)
else:
    st.warning("No Ollama models found. Please pull a model first:")
    st.code("ollama pull llama2")
    selected_chat_model = st.text_input("Enter model name manually:", value="llama2")

# Embedding model selection
embedding_models = get_available_embedding_models()
selected_embedding_model = st.selectbox(
    "Select Embedding Model", 
    embedding_models, 
    index=0,
    help="These are sentence-transformer models that will be downloaded on first use"
)

urls_input = st.text_area("Enter URLs (one per line) to ingest for Q&A")

INDEX_PATH = "faiss_index_offline"

def load_index(embedding_model_name):
    embedding_function = LocalEmbeddings(model_name=embedding_model_name)
    return FAISS.load_local(INDEX_PATH, embedding_function, allow_dangerous_deserialization=True)

# Initialize session state
if "embedding_model_name" not in st.session_state:
    st.session_state.embedding_model_name = selected_embedding_model

# On ingest button click
if st.button("Ingest URLs and Build Index"):
    urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
    if not urls:
        st.warning("Please enter at least one URL.")
        st.stop()

    all_texts = []
    for url in urls:
        st.write(f"Extracting text from {url}")
        text = extract_text_from_url(url)
        if text:
            all_texts.append(text)

    if not all_texts:
        st.error("No text extracted from the URLs.")
        st.stop()

    combined_text = "\n\n".join(all_texts)
    st.session_state.combined_text = combined_text

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(combined_text)
    st.write(f"Split text into {len(chunks)} chunks.")

    # Initialize embedding model
    with st.spinner(f"Loading embedding model '{selected_embedding_model}'..."):
        embedding_function = LocalEmbeddings(model_name=selected_embedding_model)

    with st.spinner("Generating embeddings and building index..."):
        vectorstore = FAISS.from_texts(chunks, embedding_function)

        if not os.path.exists(INDEX_PATH):
            os.makedirs(INDEX_PATH)
        vectorstore.save_local(INDEX_PATH)

    st.session_state.vectorstore = vectorstore
    st.session_state.chat_model = selected_chat_model
    st.session_state.embedding_model_name = selected_embedding_model
    st.success("Index built and saved successfully! You can now ask questions.")

# Auto-load saved index if exists and not in session state
if "vectorstore" not in st.session_state:
    if os.path.exists(INDEX_PATH):
        try:
            st.session_state.vectorstore = load_index(st.session_state.embedding_model_name)
            st.session_state.chat_model = selected_chat_model
            st.success("Loaded previously saved index! You can ask questions directly.")
        except Exception as e:
            st.warning(f"Failed to load saved index: {e}")

# Question answering UI
if "vectorstore" in st.session_state and "chat_model" in st.session_state:
    query = st.text_input("Ask a question based on ingested content:")

    if st.button("Get Answer") and query.strip():
        with st.spinner("Retrieving answer..."):
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 3}
            )
            relevant_docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            prompt = (
                "Answer the question thoroughly and provide a detailed explanation based ONLY on the following context:\n\n"
                f"{context}\n\nQuestion: {query}\nAnswer with detailed explanation:"
            )

            # Use local LLM
            llm = LocalLLM(model_name=st.session_state.chat_model)
            answer = llm._call(prompt)

            st.subheader("Answer:")
            st.write(answer)

            # Show retrieved context in an expander
            with st.expander("View Retrieved Context"):
                st.text(context)

else:
    st.info("Please ingest URLs and build the index first.")

# Add installation instructions
with st.sidebar:
    st.header("Setup Instructions")
    st.markdown("""
    ### Prerequisites:
    
    1. **Install Ollama**:
       - Visit [ollama.ai](https://ollama.ai)
       - Download and install for your OS
    
    2. **Start Ollama**:
       ```bash
       ollama serve
       ```
    
    3. **Pull a model**:
       ```bash
       ollama pull llama2
       # or other models like:
       # ollama pull mistral
       # ollama pull phi
       ```
    
    4. **Install Python dependencies**:
       ```bash
       pip install sentence-transformers
       pip install langchain-community
       ```
    
    ### Status:
    """)
    
    if ollama_running:
        st.success("✅ Ollama is running")
        if ollama_models:
            st.info(f"Available models: {len(ollama_models)}")
    else:
        st.error("❌ Ollama is not running")