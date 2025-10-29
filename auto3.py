import streamlit as st
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from typing import Optional, List, Dict, Any
import urllib3
import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModel,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
import json
import threading
import queue
from pydantic import Field
import re
import time

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContentExtractor:
    """Enhanced content extraction from web pages including dynamic content"""
    
    def __init__(self):
        self.setup_selenium()
    
    def setup_selenium(self):
        """Setup Selenium with Chrome driver"""
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")  # Run in background
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
    
    def extract_with_selenium(self, url: str) -> str:
        """Extract content using Selenium for dynamic pages"""
        driver = None
        try:
            # Create driver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.chrome_options)
            
            # Load the page
            driver.get(url)
            
            # Wait for content to load
            wait = WebDriverWait(driver, 10)
            
            # Try to wait for main content areas
            content_selectors = [
                "main",
                "article", 
                "#main",
                ".main-content",
                "[role='main']",
                "#content",
                ".content",
                ".tutorial-content",
                ".w3-main"  # W3Schools specific
            ]
            
            content_element = None
            for selector in content_selectors:
                try:
                    content_element = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    if content_element:
                        break
                except:
                    continue
            
            # If no specific content area found, wait a bit for page to fully load
            if not content_element:
                time.sleep(3)
            
            # Get the page source after JavaScript execution
            page_source = driver.page_source
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(page_source, "html.parser")
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                               'aside', 'form', 'button', 'input', 'meta', 'link',
                               'noscript', 'iframe']):
                element.decompose()
            
            # Remove ads and social elements
            for element in soup.find_all(class_=re.compile(r'(ad|ads|advertisement|social|share|cookie|banner)', re.I)):
                element.decompose()
            
            # Try to find main content
            main_content = None
            
            # W3Schools specific selectors
            w3_selectors = [
                {'id': 'main'},
                {'class': 'w3-main'},
                {'class': 'w3-container'},
                {'id': 'belowtopnav'}
            ]
            
            for selector in w3_selectors:
                main_content = soup.find('div', selector)
                if main_content:
                    break
            
            # General content selectors
            if not main_content:
                for tag in ['main', 'article', 'section']:
                    main_content = soup.find(tag)
                    if main_content:
                        break
            
            # Extract text
            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                # Fallback to body content
                body = soup.find('body')
                if body:
                    text = body.get_text(separator="\n", strip=True)
                else:
                    text = soup.get_text(separator="\n", strip=True)
            
            # Clean up the text
            lines = []
            for line in text.splitlines():
                line = line.strip()
                # Filter out navigation and non-content
                if (line and 
                    len(line) > 10 and
                    not line.startswith(('❮', '❯', '×')) and
                    not re.match(r'^(Previous|Next|Home|Tutorial|Reference|Examples|Exercises)$', line, re.I) and
                    'ADVERTISEMENT' not in line and
                    'color picker' not in line.lower()):
                    lines.append(line)
            
            final_text = '\n\n'.join(lines)
            
            return final_text
            
        except Exception as e:
            st.error(f"Selenium extraction error for {url}: {e}")
            return ""
        finally:
            if driver:
                driver.quit()
    
    def extract_with_requests(self, url: str) -> str:
        """Fallback extraction using requests"""
        try:
            headers = {
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/114.0.0.0 Safari/537.36")
            }
            res = requests.get(url, headers=headers, timeout=10, verify=False)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            
            # Similar extraction logic as above
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            text = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip() and len(line.strip()) > 20]
            
            return '\n\n'.join(lines)
            
        except Exception as e:
            st.error(f"Requests extraction error for {url}: {e}")
            return ""
    
    def extract_text_from_url(self, url: str, use_selenium: bool = True) -> str:
        """Main extraction method"""
        if use_selenium:
            text = self.extract_with_selenium(url)
            if not text or len(text) < 100:
                # Fallback to requests
                st.warning("Selenium extraction failed, trying fallback method...")
                text = self.extract_with_requests(url)
        else:
            text = self.extract_with_requests(url)
        
        if not text:
            st.error(f"Could not extract content from {url}")
        elif len(text) < 100:
            st.warning(f"Very little content extracted from {url}")
        
        return text

# [Keep all the other classes - QADataset, TrainableEmbeddings, TrainableLLM - exactly the same as before]

class QADataset(Dataset):
    """Dataset for training QA model"""
    def __init__(self, contexts, questions, answers, tokenizer, max_length=512):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        context = self.contexts[idx]
        question = self.questions[idx]
        answer = self.answers[idx]
        
        input_text = f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        target_text = answer
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }

class TrainableEmbeddings(Embeddings):
    """Embeddings that can be fine-tuned on domain data"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        with torch.no_grad():
            for text in texts:
                encoded = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
                model_output = self.model(**encoded)
                sentence_embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                embeddings.append(sentence_embeddings.cpu().numpy()[0].tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class TrainableLLM(LLM):
    """LLM that can be fine-tuned on QA pairs"""
    
    model_name: str = Field(default="google/flan-t5-base")
    training_enabled: bool = Field(default=True)
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)
    training_queue: Any = Field(default=None, exclude=True)
    training_data: Dict = Field(default_factory=lambda: {"contexts": [], "questions": [], "answers": []})
    training_thread: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(DEVICE)
        self.training_queue = queue.Queue()
        
        if self.training_enabled:
            self.training_thread = threading.Thread(target=self._background_training, daemon=True)
            self.training_thread.start()
    
    @property
    def _llm_type(self) -> str:
        return "trainable_llm"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt = prompt.strip()
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=300,
                min_length=20,
                num_beams=5,
                temperature=0.8,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.3,
                length_penalty=1.2,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.strip()
        
        if not response or len(response) < 10:
            response = "I couldn't find specific information about that in the provided context."
        
        return response
    
    def add_training_example(self, context: str, question: str, answer: str):
        if self.training_enabled:
            self.training_queue.put({
                "context": context,
                "question": question,
                "answer": answer
            })
    
    def _background_training(self):
        while True:
            examples = []
            while not self.training_queue.empty() and len(examples) < 16:
                try:
                    examples.append(self.training_queue.get(timeout=1))
                except:
                    break
            
            if len(examples) >= 4:
                for ex in examples:
                    self.training_data["contexts"].append(ex["context"])
                    self.training_data["questions"].append(ex["question"])
                    self.training_data["answers"].append(ex["answer"])
                
                self._train_on_batch()
            
            if not examples:
                import time
                time.sleep(10)
    
    def _train_on_batch(self):
        if len(self.training_data["contexts"]) < 4:
            return
        
        recent_idx = min(50, len(self.training_data["contexts"]))
        dataset = QADataset(
            contexts=self.training_data["contexts"][-recent_idx:],
            questions=self.training_data["questions"][-recent_idx:],
            answers=self.training_data["answers"][-recent_idx:],
            tokenizer=self.tokenizer
        )
        
        training_args = TrainingArguments(
            output_dir="./model_checkpoints",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            warmup_steps=20,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",
            learning_rate=3e-5,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        try:
            trainer.train()
            print(f"Model updated with {len(dataset)} examples")
        except Exception as e:
            print(f"Training error: {e}")
    
    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        with open(os.path.join(path, "training_data.json"), "w") as f:
            json.dump(self.training_data, f)
    
    def load_model(self, path: str):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        training_data_path = os.path.join(path, "training_data.json")
        if os.path.exists(training_data_path):
            with open(training_data_path, "r") as f:
                self.training_data = json.load(f)

# Streamlit UI
st.title("Self-Training Contextual Q&A System")

# Model selection
st.subheader("Model Configuration")

model_options = {
    "google/flan-t5-base": "FLAN-T5 Base (Best for Q&A, 220M params)",
    "google/flan-t5-small": "FLAN-T5 Small (Faster, 60M params)",
    "t5-base": "T5 Base (Original, 220M params)",
    "t5-small": "T5 Small (Original, 60M params)"
}

embedding_options = {
    "sentence-transformers/all-mpnet-base-v2": "MPNet (Best quality)",
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM (Fast, good quality)",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "Multilingual MiniLM"
}

selected_model = st.selectbox(
    "Select Language Model",
    list(model_options.keys()),
    format_func=lambda x: f"{x} - {model_options[x]}",
    index=0
)

selected_embedding_model = st.selectbox(
    "Select Embedding Model",
    list(embedding_options.keys()),
    format_func=lambda x: f"{x} - {embedding_options[x]}"
)

enable_training = st.checkbox("Enable Continuous Learning", value=True)
use_selenium = st.checkbox("Use Selenium for Dynamic Pages (Required for W3Schools)", value=True)
show_debug = st.checkbox("Show Debug Information", value=True)

# Initialize models
@st.cache_resource
def init_models(model_name, embedding_model_name, training_enabled):
    llm = TrainableLLM(model_name=model_name, training_enabled=training_enabled)
    embeddings = TrainableEmbeddings(model_name=embedding_model_name)
    return llm, embeddings

with st.spinner("Loading models..."):
    llm, embeddings = init_models(selected_model, selected_embedding_model, enable_training)

# URL input
urls_input = st.text_area("Enter URLs (one per line) to ingest for Q&A")

INDEX_PATH = "faiss_index_trainable"
MODEL_PATH = "fine_tuned_model"

# Content extractor
@st.cache_resource
def get_extractor():
    return ContentExtractor()

extractor = get_extractor()

# Ingest button
if st.button("Ingest URLs and Build Index"):
    urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
    if not urls:
        st.warning("Please enter at least one URL.")
        st.stop()

    all_texts = []
    
    for url in urls:
        with st.spinner(f"Extracting content from {url}..."):
            text = extractor.extract_text_from_url(url, use_selenium=use_selenium)
            if text:
                all_texts.append(text)
                st.success(f"Extracted {len(text)} characters from {url}")
                
                # Show preview
                with st.expander(f"Preview content from {url}"):
                    st.text(text[:1000] + "..." if len(text) > 1000 else text)

    if not all_texts:
        st.error("No text extracted from the URLs.")
        st.stop()

    combined_text = "\n\n".join(all_texts)
    
    # Create chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(combined_text)
    chunks = [chunk for chunk in chunks if len(chunk) > 50]
    
    st.write(f"Split text into {len(chunks)} chunks.")

    with st.spinner("Building vector index..."):
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        if not os.path.exists(INDEX_PATH):
            os.makedirs(INDEX_PATH)
        vectorstore.save_local(INDEX_PATH)

    st.session_state.vectorstore = vectorstore
    st.session_state.chunks_preview = chunks[:5]
    st.success("Index built successfully!")

# Load saved index
if "vectorstore" not in st.session_state and os.path.exists(INDEX_PATH):
    try:
        st.session_state.vectorstore = FAISS.load_local(
            INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        st.success("Loaded saved index!")
    except Exception as e:
        st.warning(f"Failed to load saved index: {e}")

# Q&A Interface
if "vectorstore" in st.session_state:
    st.subheader("Ask Questions")
    
    query = st.text_input("Enter your question:")
    
    with st.expander("Advanced Options"):
        num_chunks = st.slider("Number of context chunks to retrieve", 1, 10, 5)
        search_type = st.selectbox("Search Type", ["similarity", "mmr"])
    
    if st.button("Get Answer", type="primary"):
        if query.strip():
            with st.spinner("Searching and generating answer..."):
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type=search_type,
                    search_kwargs={"k": num_chunks}
                )
                relevant_docs = retriever.get_relevant_documents(query)
                
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                prompt = f"""You are a helpful assistant. Answer the following question based only on the provided context.

Context:
{context[:2500]}

Question: {query}

Provide a clear and detailed answer:"""
                
                answer = llm._call(prompt)
                
                st.subheader("Answer:")
                st.write(answer)
                
                if show_debug:
                    with st.expander("Debug Information"):
                        st.write(f"**Number of chunks retrieved:** {len(relevant_docs)}")
                        for i, doc in enumerate(relevant_docs[:3]):
                            st.write(f"\n**Chunk {i+1}:**")
                            st.text(doc.page_content[:400] + "...")

# Sidebar
with st.sidebar:
    st.header("Installation Requirements")
    st.code("""
    pip install selenium
    pip install webdriver-manager
    pip install beautifulsoup4
    pip install torch transformers
    pip install sentence-transformers
    pip install langchain faiss-cpu
    pip install streamlit
    """)
    
    st.header("Tips")
    st.markdown("""
    - Enable Selenium for dynamic sites like W3Schools
    - Chrome/Chromium must be installed on your system
    - First run will download ChromeDriver automatically
    - May take longer but extracts JavaScript-rendered content
    """)