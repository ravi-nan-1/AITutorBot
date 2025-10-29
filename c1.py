import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import re

# -----------------------------
# Extractors
# -----------------------------
def extract_web(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    return soup.get_text()

def extract_pdf(file):
    text = ""
    pdf = PyPDF2.PdfReader(file)
    for page in pdf.pages:
        text += page.extract_text()
    return text

def extract_youtube_transcript(url):
    # Extract video ID
    video_id = None
    match = re.search(r"(?:v=|youtu\.be/)([\w-]+)", url)
    if match:
        video_id = match.group(1)
    if not video_id:
        return "Could not extract transcript (invalid link)"
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript])
    except:
        return "Transcript not available for this video."

# -----------------------------
# Vectorstore Builder
# -----------------------------
def build_vectorstore(docs, db_path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_text(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(splits, embeddings)
    vectordb.save_local(db_path)
    return vectordb

def load_vectorstore(db_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

# -----------------------------
# Tutor QA
# -----------------------------
def tutor_qa(question, vectordb):
    llm = Ollama(model="mistral")  # You can change to llama2, phi, etc.

    # Strict prompt (don’t hallucinate outside context)
    prompt_template = """
    You are an AI tutor. Use ONLY the provided context to answer the question.
    If the answer is not in the context, say:
    "I could not find the answer in the provided material."

    Context:
    {context}

    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa.run(question)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="AI Tutor App", layout="wide")
st.title("📘 Offline AI Tutor")

st.sidebar.header("📥 Add Knowledge Sources")
content = ""

# Upload PDF
pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if pdf_file:
    st.sidebar.success("PDF uploaded ✅")
    content += extract_pdf(pdf_file)

# Web URL
web_url = st.sidebar.text_input("Enter Web Page URL")
if web_url:
    try:
        content += extract_web(web_url)
        st.sidebar.success("Web page extracted ✅")
    except:
        st.sidebar.error("Failed to fetch web page.")

# YouTube URL
yt_url = st.sidebar.text_input("Enter YouTube Video URL")
if yt_url:
    yt_text = extract_youtube_transcript(yt_url)
    if "Transcript not available" not in yt_text:
        st.sidebar.success("YouTube transcript extracted ✅")
        content += yt_text
    else:
        st.sidebar.warning("Transcript not available.")

# Build Knowledge Base Button
if st.sidebar.button("📚 Build Tutor Knowledge Base"):
    if content.strip():
        build_vectorstore(content)
        st.sidebar.success("Knowledge Base built! You can now chat with Tutor.")
    else:
        st.sidebar.error("No content provided.")

# Chat Section
st.subheader("💬 Chat with your AI Tutor")

if os.path.exists("faiss_index"):
    vectordb = load_vectorstore("faiss_index")
    question = st.text_input("Ask a question:")
    if st.button("Ask"):
        if question.strip():
            answer = tutor_qa(question, vectordb)
            st.markdown(f"**Tutor:** {answer}")
else:
    st.info("👉 Upload content and build knowledge base to start chatting.")
