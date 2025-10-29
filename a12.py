import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from typing import Optional, List, Dict, Any
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from pydantic import Field
import re
import time
from urllib.parse import urljoin, urlparse
import gc
import tempfile
from datetime import datetime

# Configuration
CACHE_DIR = os.path.join(tempfile.gettempdir(), "tutor_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cpu")

# ========== WEB CRAWLER ==========
class TutorWebCrawler:
    def __init__(self, max_pages=15):
        self.max_pages = max_pages
        self.visited_urls = set()
        self.scraped_data = []
        
    def extract_content(self, url):
        """Extract educational content from URL"""
        try:
            response = requests.get(url, timeout=10, verify=False,
                                  headers={'User-Agent': 'Mozilla/5.0'})
            
            if len(response.content) > 2097152:  # 2MB limit
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove non-content elements
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            title = soup.find('title')
            title_text = title.get_text() if title else "No Title"
            
            # Extract main content
            main_content = soup.get_text(separator='\n', strip=True)
            
            # Clean and limit text
            main_content = ' '.join(main_content.split())[:15000]
            
            return {
                'url': url,
                'title': title_text[:200],
                'content': main_content,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return None
    
    def crawl_website(self, start_url):
        """Crawl website for educational content"""
        urls_to_visit = [start_url]
        base_domain = urlparse(start_url).netloc
        
        while urls_to_visit and len(self.scraped_data) < self.max_pages:
            url = urls_to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
                
            self.visited_urls.add(url)
            data = self.extract_content(url)
            
            if data and len(data['content']) > 100:
                self.scraped_data.append(data)
                
                # Extract links for deeper crawling
                try:
                    response = requests.get(url, timeout=5)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    for link in soup.find_all('a', href=True)[:10]:  # Limit links
                        next_url = urljoin(url, link['href'])
                        if base_domain in next_url and next_url not in self.visited_urls:
                            urls_to_visit.append(next_url)
                except:
                    pass
            
            time.sleep(0.3)
            
        return self.scraped_data

# ========== EMBEDDINGS ==========
class TutorEmbeddings(Embeddings):
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
            self.model = AutoModel.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
            self.model.eval()
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
    
    @torch.no_grad()
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text[:512], return_tensors="pt", 
                                   truncation=True, max_length=512, padding=True)
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            embeddings.append(embedding)
        gc.collect()
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# ========== FIXED TUTOR LLM ==========
class TutorLLM(LLM):
    model_name: str = Field(default="google/flan-t5-base")
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
            self.model.eval()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    
    @property
    def _llm_type(self) -> str:
        return "tutor_llm"
    
    @torch.no_grad()
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # Clean and truncate prompt
            prompt = prompt.replace('\n\n', '\n').strip()
            prompt = prompt[:1024]
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding=True
            )
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                min_length=30,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean response
            response = response.strip()
            if not response or len(response) < 10:
                response = "I need more context to provide a detailed answer. Could you please be more specific?"
            
            gc.collect()
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an issue generating a response. Please try rephrasing your question."

# ========== SIMPLIFIED RAG TUTOR ==========
class RAGTutor:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.conversation_history = []
    
    def create_prompt(self, question: str, context: str, mode: str, difficulty: str) -> str:
        """Create a simple, effective prompt"""
        
        # Simple mode-based prompts
        if mode == "explain":
            if difficulty == "beginner":
                return f"Explain simply: {question}\nUsing this information: {context[:800]}\nSimple explanation:"
            elif difficulty == "advanced":
                return f"Provide detailed analysis of: {question}\nBased on: {context[:800]}\nDetailed explanation:"
            else:
                return f"Explain: {question}\nContext: {context[:800]}\nExplanation:"
        
        elif mode == "simplify":
            return f"Simplify this concept: {question}\nInformation: {context[:600]}\nSimple version:"
        
        elif mode == "elaborate":
            return f"Elaborate on: {question}\nDetails: {context[:800]}\nFull elaboration:"
        
        elif mode == "summarize":
            return f"Summarize: {question}\nContent: {context[:600]}\nSummary:"
        
        elif mode == "quiz":
            return f"Create 3 quiz questions about: {question}\nBased on: {context[:600]}\nQuestions:"
        
        else:
            return f"Answer: {question}\nUsing: {context[:800]}\nAnswer:"
    
    def get_response(self, question: str, mode: str = "explain", difficulty: str = "medium"):
        """Get tutoring response"""
        
        try:
            # Search for relevant documents
            docs = self.vectorstore.similarity_search(question, k=4)
            
            if not docs:
                return {
                    "answer": "I don't have enough information about this topic. Please make sure you've loaded relevant content.",
                    "sources": [],
                    "suggestions": ["Try loading more content", "Ask a different question", "Be more specific"]
                }
            
            # Combine context from documents
            context = " ".join([doc.page_content[:200] for doc in docs])
            
            # Create simple prompt
            prompt = self.create_prompt(question, context, mode.lower(), difficulty.lower())
            
            # Get response from LLM
            response = self.llm._call(prompt)
            
            # Store in history
            self.conversation_history.append({
                "question": question,
                "answer": response,
                "mode": mode,
                "timestamp": datetime.now()
            })
            
            # Extract sources
            sources = []
            for doc in docs[:3]:
                sources.append({
                    "title": doc.metadata.get("title", "Unknown"),
                    "url": doc.metadata.get("url", "#")
                })
            
            # Generate suggestions
            suggestions = self.generate_suggestions(question, mode)
            
            return {
                "answer": response,
                "sources": sources,
                "suggestions": suggestions
            }
            
        except Exception as e:
            return {
                "answer": f"I encountered an error. Please try again with a different question.",
                "sources": [],
                "suggestions": ["Try a simpler question", "Reload the content", "Clear cache and restart"]
            }
    
    def generate_suggestions(self, question: str, mode: str) -> List[str]:
        """Generate follow-up suggestions"""
        base_suggestions = [
            "Tell me more",
            "Give an example",
            "Explain differently",
            "Quiz me on this"
        ]
        
        if mode == "explain":
            return ["Simplify this", "Give examples", "More details", "Related topics"]
        elif mode == "quiz":
            return ["Check my answer", "More questions", "Explain the answers", "Different topic"]
        else:
            return base_suggestions[:3]

# ========== STREAMLIT UI WITH FIXED CHAT LAYOUT ==========
st.set_page_config(page_title="AI Tutor Assistant", layout="wide", page_icon="🎓")

# Custom CSS for chat interface
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        bottom: 20px;
        background-color: white;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stChatMessage):last-of-type {
        padding-bottom: 5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 AI Tutor - Your Learning Assistant")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Learning Mode
    learning_mode = st.selectbox(
        "Learning Mode:",
        ["Explain", "Simplify", "Elaborate", "Summarize", "Quiz"]
    )
    
    # Difficulty
    difficulty = st.selectbox(
        "Difficulty:",
        ["Beginner", "Medium", "Advanced"]
    )
    
    # Pages to crawl
    max_pages = st.slider("Pages to analyze:", 5, 20, 10)
    
    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat"):
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Clear All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Main tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat & Learn", "📚 Load Content", "📝 Practice"])

# ========== TAB 2: LOAD CONTENT (Moved before chat for better flow) ==========
with tab2:
    st.header("📚 Load Learning Content")
    
    url_input = st.text_input(
        "Enter URL to learn from:",
        placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence"
    )
    
    # Quick examples
    st.markdown("**Quick Examples:**")
    col1, col2, col3 = st.columns(3)
    
    examples = [
        ("AI & ML", "https://en.wikipedia.org/wiki/Machine_learning"),
        ("Biology", "https://en.wikipedia.org/wiki/Cell_biology"),
        ("Physics", "https://en.wikipedia.org/wiki/Physics")
    ]
    
    for col, (label, url) in zip([col1, col2, col3], examples):
        with col:
            if st.button(label, key=f"ex_{label}"):
                url_input = url
    
    if st.button("🔍 Load Content", type="primary"):
        if url_input:
            with st.spinner("Loading content..."):
                # Crawl
                crawler = TutorWebCrawler(max_pages=max_pages)
                scraped_data = crawler.crawl_website(url_input)
                
                if scraped_data:
                    # Create embeddings
                    embeddings = TutorEmbeddings()
                    
                    # Process text
                    texts = []
                    metadata = []
                    
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    
                    for page in scraped_data:
                        chunks = splitter.split_text(page['content'])
                        for chunk in chunks[:10]:
                            texts.append(chunk)
                            metadata.append({
                                'url': page['url'],
                                'title': page['title']
                            })
                    
                    # Create vector store
                    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
                    st.session_state.vectorstore = vectorstore
                    
                    # Initialize LLM and Tutor
                    if "llm" not in st.session_state:
                        st.session_state.llm = TutorLLM()
                    
                    st.session_state.tutor = RAGTutor(vectorstore, st.session_state.llm)
                    st.session_state.content_loaded = True
                    st.session_state.content_info = {
                        'url': url_input,
                        'pages': len(scraped_data),
                        'chunks': len(texts)
                    }
                    
                    st.success(f"✅ Loaded {len(scraped_data)} pages with {len(texts)} chunks!")
                    st.info("Go to 'Chat & Learn' tab to start learning!")
                else:
                    st.error("Could not load content. Please try another URL.")

# ========== TAB 1: CHAT INTERFACE ==========
with tab1:
    if "content_loaded" not in st.session_state:
        st.info("👈 Please load content from the 'Load Content' tab first")
        st.stop()
    
    # Show content info
    info = st.session_state.get("content_info", {})
    st.success(f"📖 Learning from: {info.get('url', 'Unknown')}")
    
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your AI tutor. I've studied the content you loaded. What would you like to learn about?"
        })
    
    # Create a container for messages that will scroll
    messages_container = st.container()
    
    # Display all messages in the scrollable container
    with messages_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show sources if available
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"• {source['title']}")
    
    # Show suggestions if available
    if len(st.session_state.messages) > 0:
        last_msg = st.session_state.messages[-1]
        if last_msg.get("suggestions") and last_msg["role"] == "assistant":
            cols = st.columns(len(last_msg["suggestions"]))
            for col, suggestion in zip(cols, last_msg["suggestions"]):
                with col:
                    if st.button(suggestion, key=f"sug_{len(st.session_state.messages)}_{suggestion[:10]}"):
                        st.session_state.pending_question = suggestion
                        st.rerun()
    
    # Chat input at the bottom
    if "pending_question" in st.session_state:
        prompt = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        prompt = st.chat_input("Ask me anything about the loaded content...")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with messages_container:
            with st.chat_message("user"):
                st.write(prompt)
        
        # Get and display assistant response
        with messages_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get response from tutor
                    tutor = st.session_state.tutor
                    response = tutor.get_response(
                        prompt,
                        mode=learning_mode.lower(),
                        difficulty=difficulty.lower()
                    )
                    
                    # Display response
                    st.write(response["answer"])
                    
                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                        "suggestions": response.get("suggestions", [])
                    })
        
        # Rerun to update the UI
        st.rerun()

# ========== TAB 3: PRACTICE ==========
with tab3:
    st.header("📝 Practice & Test")
    
    if "content_loaded" not in st.session_state:
        st.info("Please load content first")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Generate Quiz")
        quiz_topic = st.text_input("Topic for quiz (optional):")
        
        if st.button("Generate Quiz"):
            with st.spinner("Creating quiz..."):
                tutor = st.session_state.tutor
                response = tutor.get_response(
                    quiz_topic or "Create a quiz about the main topics",
                    mode="quiz"
                )
                st.markdown("### Quiz Questions:")
                st.write(response["answer"])
                st.session_state.last_quiz = response["answer"]
    
    with col2:
        st.subheader("✍️ Answer Questions")
        
        if "last_quiz" in st.session_state:
            st.info("Answer the quiz questions above")
            
        your_answer = st.text_area("Your answer:")
        
        if st.button("Check Understanding"):
            if your_answer:
                # Simple feedback
                st.success("Answer submitted!")
                st.info("Review tip: Compare your answer with the source material to verify accuracy.")
            else:
                st.warning("Please write an answer first")

# Footer
st.markdown("---")
st.caption("🎓 AI Tutor - RAG-Powered Learning Assistant")