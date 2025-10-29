import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from typing import Optional, List, Dict
import urllib3
import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModel,
    TrainingArguments,
    Trainer,
    pipeline
)
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
import threading
import queue

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        # For T5/BART models
        input_text = f"question: {question} context: {context}"
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
            max_length=self.max_length,
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
                encoded = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
                model_output = self.model(**encoded)
                sentence_embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
                embeddings.append(sentence_embeddings.cpu().numpy()[0].tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class TrainableLLM(LLM):
    """LLM that can be fine-tuned on QA pairs"""
    
    def __init__(self, model_name: str = "google/flan-t5-small", training_enabled: bool = True):
        super().__init__()
        self.model_name = model_name
        self.training_enabled = training_enabled
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        
        # Training data queue
        self.training_queue = queue.Queue()
        self.training_data = {"contexts": [], "questions": [], "answers": []}
        
        # Start background training thread if enabled
        if training_enabled:
            self.training_thread = threading.Thread(target=self._background_training, daemon=True)
            self.training_thread.start()
    
    @property
    def _llm_type(self) -> str:
        return "trainable_llm"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=300,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def add_training_example(self, context: str, question: str, answer: str):
        """Add a training example to the queue"""
        if self.training_enabled:
            self.training_queue.put({
                "context": context,
                "question": question,
                "answer": answer
            })
    
    def _background_training(self):
        """Background thread for continuous training"""
        while True:
            # Collect training examples
            examples = []
            while not self.training_queue.empty() and len(examples) < 16:  # batch size
                try:
                    examples.append(self.training_queue.get(timeout=1))
                except:
                    break
            
            if len(examples) >= 8:  # Minimum batch size for training
                # Add to training data
                for ex in examples:
                    self.training_data["contexts"].append(ex["context"])
                    self.training_data["questions"].append(ex["question"])
                    self.training_data["answers"].append(ex["answer"])
                
                # Perform training
                self._train_on_batch()
            
            # Sleep if no data
            if not examples:
                import time
                time.sleep(10)
    
    def _train_on_batch(self):
        """Train the model on collected data"""
        if len(self.training_data["contexts"]) < 8:
            return
        
        # Create dataset from recent data (last 100 examples)
        recent_idx = min(100, len(self.training_data["contexts"]))
        dataset = QADataset(
            contexts=self.training_data["contexts"][-recent_idx:],
            questions=self.training_data["questions"][-recent_idx:],
            answers=self.training_data["answers"][-recent_idx:],
            tokenizer=self.tokenizer
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./model_checkpoints",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            warmup_steps=50,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",
            learning_rate=5e-5,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        try:
            trainer.train()
            st.sidebar.success(f"Model updated with {len(dataset)} examples")
        except Exception as e:
            st.sidebar.error(f"Training error: {e}")
    
    def save_model(self, path: str):
        """Save the fine-tuned model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save training data
        with open(os.path.join(path, "training_data.json"), "w") as f:
            json.dump(self.training_data, f)
    
    def load_model(self, path: str):
        """Load a fine-tuned model"""
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load training data
        training_data_path = os.path.join(path, "training_data.json")
        if os.path.exists(training_data_path):
            with open(training_data_path, "r") as f:
                self.training_data = json.load(f)

# Helper functions
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

def generate_training_pairs(text: str, num_pairs: int = 5) -> List[Dict[str, str]]:
    """Generate synthetic Q&A pairs from text for training"""
    # Simple heuristic: create questions from sentences
    sentences = text.split('.')[:num_pairs * 2]
    pairs = []
    
    for i, sentence in enumerate(sentences):
        if len(sentence.strip()) > 20:
            # Create a simple question
            if i % 2 == 0:
                question = f"What is mentioned about {sentence.strip()[:30]}...?"
            else:
                question = f"Can you explain the part about {sentence.strip()[:30]}...?"
            
            pairs.append({
                "context": text[:1000],  # Use first 1000 chars as context
                "question": question,
                "answer": sentence.strip()
            })
    
    return pairs[:num_pairs]

# Streamlit UI
st.title("Self-Training Contextual Q&A System")

# Model selection
st.subheader("Model Configuration")

model_options = {
    "google/flan-t5-small": "T5 Small (Fast, 60M params)",
    "google/flan-t5-base": "T5 Base (Balanced, 220M params)",
    "facebook/bart-base": "BART Base (Good quality, 140M params)",
    "t5-small": "T5 Small (Original, 60M params)"
}

embedding_options = {
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM (Fast, good quality)",
    "sentence-transformers/all-mpnet-base-v2": "MPNet (Best quality, slower)",
    "sentence-transformers/paraphrase-MiniLM-L6-v2": "Paraphrase MiniLM (Balanced)"
}

selected_model = st.selectbox(
    "Select Language Model",
    list(model_options.keys()),
    format_func=lambda x: f"{x} - {model_options[x]}"
)

selected_embedding_model = st.selectbox(
    "Select Embedding Model",
    list(embedding_options.keys()),
    format_func=lambda x: f"{x} - {embedding_options[x]}"
)

enable_training = st.checkbox("Enable Continuous Learning", value=True)

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

# Ingest button
if st.button("Ingest URLs and Build Index"):
    urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
    if not urls:
        st.warning("Please enter at least one URL.")
        st.stop()

    all_texts = []
    all_training_pairs = []
    
    for url in urls:
        st.write(f"Extracting text from {url}")
        text = extract_text_from_url(url)
        if text:
            all_texts.append(text)
            
            # Generate training pairs if training is enabled
            if enable_training:
                pairs = generate_training_pairs(text)
                all_training_pairs.extend(pairs)

    if not all_texts:
        st.error("No text extracted from the URLs.")
        st.stop()

    combined_text = "\n\n".join(all_texts)
    
    # Add training examples
    if enable_training and all_training_pairs:
        with st.spinner(f"Adding {len(all_training_pairs)} training examples..."):
            for pair in all_training_pairs:
                llm.add_training_example(
                    context=pair["context"],
                    question=pair["question"],
                    answer=pair["answer"]
                )

    # Create chunks and build index
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(combined_text)
    st.write(f"Split text into {len(chunks)} chunks.")

    with st.spinner("Building vector index..."):
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        if not os.path.exists(INDEX_PATH):
            os.makedirs(INDEX_PATH)
        vectorstore.save_local(INDEX_PATH)

    st.session_state.vectorstore = vectorstore
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
    
    col1, col2 = st.columns([3, 1])
    with col1:
        answer_button = st.button("Get Answer", type="primary")
    with col2:
        if enable_training:
            train_button = st.button("Train on This Q&A")
        else:
            train_button = False
    
    if answer_button and query.strip():
        with st.spinner("Searching and generating answer..."):
            # Retrieve relevant documents
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            relevant_docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Generate answer
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            answer = llm._call(prompt)
            
            # Display results
            st.subheader("Answer:")
            st.write(answer)
            
            # Store in session state for training
            st.session_state.last_qa = {
                "context": context,
                "question": query,
                "answer": answer
            }
            
            # Show context
            with st.expander("View Retrieved Context"):
                st.text(context[:1000] + "..." if len(context) > 1000 else context)
    
    # Train on specific Q&A
    if train_button and "last_qa" in st.session_state:
        qa = st.session_state.last_qa
        llm.add_training_example(
            context=qa["context"],
            question=qa["question"],
            answer=qa["answer"]
        )
        st.success("Added Q&A pair to training queue!")

# Sidebar - Model Management
with st.sidebar:
    st.header("Model Management")
    
    if st.button("Save Fine-tuned Model"):
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        llm.save_model(MODEL_PATH)
        st.success(f"Model saved to {MODEL_PATH}")
    
    if st.button("Load Fine-tuned Model"):
        if os.path.exists(MODEL_PATH):
            llm.load_model(MODEL_PATH)
            st.success("Fine-tuned model loaded!")
        else:
            st.error("No saved model found!")
    
    st.header("Training Status")
    if enable_training:
        st.info(f"Training Queue: {llm.training_queue.qsize()} examples")
        st.info(f"Total Training Data: {len(llm.training_data['questions'])} examples")
    else:
        st.warning("Training is disabled")
    
    st.header("System Info")
    st.info(f"Device: {DEVICE}")
    st.info(f"Model: {selected_model}")

# Instructions
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Self-Training Q&A System
    
    This system can automatically improve its performance by learning from:
    1. **Ingested content** - Automatically generates Q&A pairs from URLs
    2. **User interactions** - Learn from questions you ask
    
    #### Features:
    - **Continuous Learning**: Model trains in the background
    - **Save/Load Models**: Preserve your fine-tuned model
    - **Multiple Model Options**: Choose based on your needs
    
    #### Steps:
    1. Enter URLs to ingest content
    2. Ask questions about the content
    3. Optionally train on specific Q&A pairs
    4. Save your fine-tuned model for later use
    
    #### Requirements:
    ```bash
    pip install torch transformers sentence-transformers
    pip install langchain faiss-cpu streamlit
    ```
    """)