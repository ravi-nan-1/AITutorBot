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
import PyPDF2
import io
from youtube_transcript_api import YouTubeTranscriptApi
import speech_recognition as sr
from pydub import AudioSegment
import hashlib

# Configuration
CACHE_DIR = os.path.join(tempfile.gettempdir(), "tutor_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cpu")

# ========== ENHANCED WEB CRAWLER ==========
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
                'timestamp': datetime.now().isoformat(),
                'source_type': 'web'
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

# ========== CONTENT PROCESSORS ==========
class ContentProcessor:
    @staticmethod
    def process_pdf(pdf_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
            text = ""
            
            for page_num in range(min(len(pdf_reader.pages), 50)):  # Limit to 50 pages
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return {
                'title': pdf_file.name,
                'content': text,
                'source_type': 'pdf',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None
    
    @staticmethod
    def process_youtube(url):
        """Extract transcript from YouTube video"""
        try:
            # Extract video ID
            video_id = None
            if "youtube.com/watch?v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[1].split("?")[0]
            
            if not video_id:
                return None
            
            # Get transcript
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get English transcript first
            transcript = None
            for t in transcript_list:
                if t.language_code.startswith('en'):
                    transcript = t.fetch()
                    break
            
            # If no English, get any available
            if not transcript:
                transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB']).fetch()
            
            # Combine transcript text
            text = " ".join([entry['text'] for entry in transcript])
            
            return {
                'title': f"YouTube Video: {video_id}",
                'content': text,
                'source_type': 'youtube',
                'url': url,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            st.error(f"Error processing YouTube video: {e}")
            return None
    
    @staticmethod
    def process_audio(audio_file):
        """Extract text from audio file using speech recognition"""
        try:
            # Save temporary file
            temp_path = os.path.join(CACHE_DIR, f"temp_audio_{hashlib.md5(audio_file.name.encode()).hexdigest()}")
            
            with open(temp_path, "wb") as f:
                f.write(audio_file.read())
            
            # Convert to wav if needed
            audio = AudioSegment.from_file(temp_path)
            wav_path = temp_path + ".wav"
            audio.export(wav_path, format="wav")
            
            # Recognize speech
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
            
            # Clean up
            os.remove(temp_path)
            os.remove(wav_path)
            
            return {
                'title': audio_file.name,
                'content': text,
                'source_type': 'audio',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            return None

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

# ========== IMPROVED TUTOR LLM ==========
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
            # Clean and prepare prompt for better context usage
            prompt = prompt.replace('\n\n', '\n').strip()
            
            # Ensure the model focuses on the provided context
            if "Context:" in prompt or "Based on:" in prompt or "Using:" in prompt:
                prompt = f"Answer using only the given information. {prompt}"
            
            # Truncate prompt intelligently
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
                max_new_tokens=300,
                min_length=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.85,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean and enhance response
            response = response.strip()
            if not response or len(response) < 20:
                response = "Based on the provided context, I need more specific information to answer your question properly. Please provide more details or rephrase your question."
            
            gc.collect()
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an issue. Please try rephrasing your question or provide more context."

# ========== ENHANCED RAG TUTOR ==========
class RAGTutor:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.conversation_history = []
        self.all_content = []  # Store all loaded content for better context
    
    def set_all_content(self, content):
        """Store all content for generating better suggestions"""
        self.all_content = content
    
    def create_prompt(self, question: str, context: str, mode: str, difficulty: str) -> str:
        """Create enhanced prompts that force context usage"""
        
        # Increase context size for better answers
        context = context[:1500]  # Increased from 800
        
        # Add instruction to use context
        base_instruction = "Use ONLY the following information to answer. Do not use external knowledge. "
        
        if mode == "explain":
            if difficulty == "beginner":
                return f"{base_instruction}Explain in simple terms based on this context:\nContext: {context}\nQuestion: {question}\nSimple explanation based on the context:"
            elif difficulty == "advanced":
                return f"{base_instruction}Provide detailed analysis using this information:\nContext: {context}\nQuestion: {question}\nDetailed analysis from the context:"
            else:
                return f"{base_instruction}Explain using this information:\nContext: {context}\nQuestion: {question}\nExplanation from the context:"
        
        elif mode == "simplify":
            return f"{base_instruction}Simplify this based on the context:\nContext: {context}\nTopic: {question}\nSimplified version:"
        
        elif mode == "elaborate":
            return f"{base_instruction}Elaborate using this information:\nContext: {context}\nTopic: {question}\nDetailed elaboration:"
        
        elif mode == "summarize":
            return f"{base_instruction}Summarize this information:\nContext: {context}\nTopic: {question}\nSummary:"
        
        elif mode == "quiz":
            return f"{base_instruction}Create 3 quiz questions from this content:\nContext: {context}\nTopic: {question}\nQuiz questions:"
        
        else:
            return f"{base_instruction}Answer based on this context:\nContext: {context}\nQuestion: {question}\nAnswer from context:"
    
    def get_response(self, question: str, mode: str = "explain", difficulty: str = "medium"):
        """Get improved tutoring response"""
        
        try:
            # Search for more relevant documents and increase k
            docs = self.vectorstore.similarity_search(question, k=6)  # Increased from 4
            
            if not docs:
                return {
                    "answer": "I don't have information about this topic in the loaded content. Please make sure you've loaded relevant material first.",
                    "sources": [],
                    "suggestions": self.generate_dynamic_suggestions(question, [])
                }
            
            # Combine more context from documents (increased chunk size)
            context_parts = []
            for doc in docs:
                # Take more content from each document
                content = doc.page_content[:400]  # Increased from 200
                context_parts.append(content)
            
            context = " ".join(context_parts)
            
            # Create enhanced prompt
            prompt = self.create_prompt(question, context, mode.lower(), difficulty.lower())
            
            # Get response from LLM
            response = self.llm._call(prompt)
            
            # Verify response is using context
            if len(response) < 30:
                # Try again with simpler prompt
                simple_prompt = f"Based on this text: {context[:800]}\nAnswer: {question}\nResponse:"
                response = self.llm._call(simple_prompt)
            
            # Store in history
            self.conversation_history.append({
                "question": question,
                "answer": response,
                "mode": mode,
                "context": context[:500],
                "timestamp": datetime.now()
            })
            
            # Extract sources with better metadata
            sources = []
            seen_titles = set()
            for doc in docs[:4]:  # Show more sources
                title = doc.metadata.get("title", "Unknown")
                if title not in seen_titles:
                    sources.append({
                        "title": title,
                        "url": doc.metadata.get("url", "#"),
                        "type": doc.metadata.get("source_type", "web"),
                        "snippet": doc.page_content[:100] + "..."
                    })
                    seen_titles.add(title)
            
            # Generate dynamic suggestions based on actual content
            suggestions = self.generate_dynamic_suggestions(question, docs)
            
            return {
                "answer": response,
                "sources": sources,
                "suggestions": suggestions,
                "context_used": context[:200] + "..."  # Show what context was used
            }
            
        except Exception as e:
            return {
                "answer": f"I encountered an error processing your question. Please try again.",
                "sources": [],
                "suggestions": ["Try a simpler question", "Reload the content", "Ask about a specific topic"]
            }
    
    def generate_dynamic_suggestions(self, question: str, docs: List) -> List[str]:
        """Generate suggestions based on actual content"""
        suggestions = []
        
        # Extract key topics from the documents
        if docs:
            # Get unique words from documents
            all_text = " ".join([doc.page_content[:200] for doc in docs])
            
            # Simple keyword extraction
            words = re.findall(r'\b[A-Z][a-z]+\b', all_text)
            unique_topics = list(set(words))[:3]
            
            if unique_topics:
                suggestions.append(f"Tell me more about {unique_topics[0]}")
                if len(unique_topics) > 1:
                    suggestions.append(f"Explain {unique_topics[1]}")
            
            # Mode-based suggestions
            suggestions.append("Give me an example")
            suggestions.append("Quiz me on this topic")
        else:
            # Default suggestions when no content
            suggestions = [
                "Load more content first",
                "Try a different topic",
                "Ask a general question"
            ]
        
        return suggestions[:4]  # Return maximum 4 suggestions

# ========== STREAMLIT UI WITH ENHANCED FEATURES ==========
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
    .context-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stChatMessage):last-of-type {
        padding-bottom: 5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 AI Tutor - Your Learning Assistant")

def process_loaded_content(content_data):
    """Process and vectorize loaded content"""
    try:
        # Create embeddings
        embeddings = TutorEmbeddings()
        
        # Process text
        texts = []
        metadata = []
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Increased for better context
            chunk_overlap=100  # More overlap for continuity
        )
        
        for item in content_data:
            chunks = splitter.split_text(item['content'])
            for i, chunk in enumerate(chunks[:20]):  # Increased chunk limit
                texts.append(chunk)
                metadata.append({
                    'url': item.get('url', ''),
                    'title': item.get('title', 'Unknown'),
                    'source_type': item.get('source_type', 'unknown'),
                    'chunk_index': i
                })
        
        # Create or update vector store
        if "vectorstore" in st.session_state:
            # Add to existing vectorstore
            new_vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
            st.session_state.vectorstore.merge_from(new_vectorstore)
        else:
            # Create new vectorstore
            st.session_state.vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
        
        # Initialize LLM and Tutor
        if "llm" not in st.session_state:
            st.session_state.llm = TutorLLM()
        
        st.session_state.tutor = RAGTutor(st.session_state.vectorstore, st.session_state.llm)
        st.session_state.tutor.set_all_content(st.session_state.all_loaded_content)
        st.session_state.content_loaded = True
        
        # Update content info
        if "content_info" not in st.session_state:
            st.session_state.content_info = {
                'sources': [],
                'total_chunks': 0
            }
        
        st.session_state.content_info['sources'].append({
            'type': content_data[0].get('source_type', 'unknown'),
            'title': content_data[0].get('title', 'Unknown')
        })
        st.session_state.content_info['total_chunks'] += len(texts)
        
        st.success(f"✅ Successfully loaded {len(texts)} text chunks!")
        st.info("Go to 'Chat & Learn' tab to start learning!")
        
    except Exception as e:
        st.error(f"Error processing content: {e}")


# Initialize session state
if "all_loaded_content" not in st.session_state:
    st.session_state.all_loaded_content = []

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
    
    # Show loaded content summary
    if st.session_state.all_loaded_content:
        st.markdown("### 📚 Loaded Content")
        for content in st.session_state.all_loaded_content[-3:]:  # Show last 3
            st.caption(f"• {content.get('title', 'Unknown')[:30]}... ({content.get('source_type', 'unknown')})")
    
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

# ========== TAB 2: LOAD CONTENT (Enhanced) ==========
with tab2:
    st.header("📚 Load Learning Content")
    
    # Content type selector
    content_type = st.radio(
        "Select content type:",
        ["🌐 Website", "📄 PDF", "🎥 YouTube", "🎵 Audio"],
        horizontal=True
    )
    
    content_processor = ContentProcessor()
    
    if content_type == "🌐 Website":
        url_input = st.text_input(
            "Enter URL to learn from:",
            placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence"
        )
        
        # Enhanced quick examples with more variety
        st.markdown("**Quick Examples:**")
        col1, col2, col3, col4 = st.columns(4)
        
        examples = [
            ("AI & ML", "https://en.wikipedia.org/wiki/Machine_learning"),
            ("Biology", "https://en.wikipedia.org/wiki/Cell_biology"),
            ("Physics", "https://en.wikipedia.org/wiki/Physics"),
            ("History", "https://en.wikipedia.org/wiki/World_history")
        ]
        
        for col, (label, url) in zip([col1, col2, col3, col4], examples):
            with col:
                if st.button(label, key=f"ex_{label}"):
                    url_input = url
        
        if st.button("🔍 Load Website", type="primary"):
            if url_input:
                with st.spinner("Loading website content..."):
                    crawler = TutorWebCrawler(max_pages=max_pages)
                    scraped_data = crawler.crawl_website(url_input)
                    
                    if scraped_data:
                        st.session_state.all_loaded_content.extend(scraped_data)
                        process_loaded_content(scraped_data)
    
    elif content_type == "📄 PDF":
        pdf_file = st.file_uploader("Upload PDF file", type=['pdf'])
        
        if pdf_file and st.button("📄 Load PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                pdf_data = content_processor.process_pdf(pdf_file)
                if pdf_data:
                    st.session_state.all_loaded_content.append(pdf_data)
                    process_loaded_content([pdf_data])
    
    elif content_type == "🎥 YouTube":
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        
        # YouTube examples
        st.markdown("**Popular Educational Channels:**")
        examples = [
            "Khan Academy", "MIT OpenCourseWare", 
            "CrashCourse", "TED-Ed"
        ]
        st.caption(" • ".join(examples))
        
        if youtube_url and st.button("🎥 Load Video", type="primary"):
            with st.spinner("Extracting video transcript..."):
                video_data = content_processor.process_youtube(youtube_url)
                if video_data:
                    st.session_state.all_loaded_content.append(video_data)
                    process_loaded_content([video_data])
    
    elif content_type == "🎵 Audio":
        audio_file = st.file_uploader(
            "Upload audio file", 
            type=['mp3', 'wav', 'ogg', 'm4a']
        )
        
        if audio_file and st.button("🎵 Process Audio", type="primary"):
            with st.spinner("Transcribing audio..."):
                audio_data = content_processor.process_audio(audio_file)
                if audio_data:
                    st.session_state.all_loaded_content.append(audio_data)
                    process_loaded_content([audio_data])


# ========== TAB 1: CHAT INTERFACE (Enhanced) ==========
with tab1:
    if "content_loaded" not in st.session_state:
        st.info("👈 Please load content from the 'Load Content' tab first")
        
        # Show quick start guide
        st.markdown("""
        ### 🚀 Quick Start Guide
        1. Go to **Load Content** tab
        2. Choose your content type (Website, PDF, YouTube, Audio)
        3. Load your learning material
        4. Come back here to start learning!
        
        **Supported formats:**
        - 🌐 Any educational website
        - 📄 PDF documents
        - 🎥 YouTube videos (with captions)
        - 🎵 Audio lectures (MP3, WAV, etc.)
        """)
        st.stop()
    
    # Show content info
    info = st.session_state.get("content_info", {})
    col1, col2 = st.columns([3, 1])
    with col1:
        sources_list = ", ".join([s['type'] for s in info.get('sources', [])])
        st.success(f"📖 Learning from: {sources_list} ({info.get('total_chunks', 0)} chunks)")
    with col2:
        show_context = st.checkbox("Show context", value=False)
    
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your AI tutor. I've studied the content you loaded. What would you like to learn about? You can ask me to explain, simplify, elaborate, or quiz you on any topic from the material."
        })
    
    # Create a container for messages
    messages_container = st.container()
    
    # Display all messages
    with messages_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show context if enabled
                if show_context and message.get("context_used") and message["role"] == "assistant":
                    st.markdown(f'<div class="context-box">📋 Context: {message["context_used"]}</div>', 
                              unsafe_allow_html=True)
                
                # Show sources
                if message.get("sources") and len(message["sources"]) > 0:
                    with st.expander(f"📚 Sources ({len(message['sources'])})"):
                        for source in message["sources"]:
                            st.markdown(f"**{source['title']}** ({source['type']})")
                            if source.get('snippet'):
                                st.caption(source['snippet'])
                            if source.get('url') and source['url'] != '#':
                                st.caption(f"🔗 {source['url'][:50]}...")
    
    # Show dynamic suggestions
    if len(st.session_state.messages) > 0:
        last_msg = st.session_state.messages[-1]
        if last_msg.get("suggestions") and last_msg["role"] == "assistant":
            st.markdown("**💡 Suggested questions:**")
            cols = st.columns(min(len(last_msg["suggestions"]), 4))
            for col, suggestion in zip(cols, last_msg["suggestions"]):
                with col:
                    if st.button(suggestion, key=f"sug_{len(st.session_state.messages)}_{hash(suggestion)}"):
                        st.session_state.pending_question = suggestion
                        st.rerun()
    
    # Chat input
    if "pending_question" in st.session_state:
        prompt = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        prompt = st.chat_input("Ask me anything about the loaded content...")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
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
                    
                    # Add to messages with all metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                        "suggestions": response.get("suggestions", []),
                        "context_used": response.get("context_used", "")
                    })
        
        # Rerun to update UI
        st.rerun()

# ========== TAB 3: PRACTICE (Enhanced) ==========
with tab3:
    st.header("📝 Practice & Test")
    
    if "content_loaded" not in st.session_state:
        st.info("Please load content first")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Generate Quiz")
        
        # Auto-suggest topics based on loaded content
        if st.session_state.all_loaded_content:
            st.caption("Suggested topics from your content:")
            # Extract some topic suggestions
            sample_content = st.session_state.all_loaded_content[0]['content'][:500]
            topics = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sample_content)
            if topics:
                st.caption(" • ".join(topics[:3]))
        
        quiz_topic = st.text_input("Topic for quiz (optional - leave empty for general):")
        quiz_difficulty = st.select_slider(
            "Quiz difficulty:",
            options=["Easy", "Medium", "Hard"],
            value="Medium"
        )
        
        if st.button("Generate Quiz", type="primary"):
            with st.spinner("Creating quiz..."):
                tutor = st.session_state.tutor
                quiz_prompt = quiz_topic if quiz_topic else "the main topics in the loaded content"
                response = tutor.get_response(
                    f"Create a {quiz_difficulty.lower()} quiz about {quiz_prompt}",
                    mode="quiz",
                    difficulty=quiz_difficulty.lower()
                )
                st.markdown("### 📋 Quiz Questions:")
                st.write(response["answer"])
                st.session_state.last_quiz = response["answer"]
                
                # Show sources for quiz
                if response.get("sources"):
                    with st.expander("📚 Quiz based on these sources"):
                        for source in response["sources"]:
                            st.caption(f"• {source['title']}")
    
    with col2:
        st.subheader("✍️ Answer & Feedback")
        
        if "last_quiz" in st.session_state:
            st.info("Answer the quiz questions from the left panel")
            
        your_answer = st.text_area("Your answer:", height=150)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Check Understanding"):
                if your_answer:
                    with st.spinner("Analyzing your answer..."):
                        # Get feedback on the answer
                        tutor = st.session_state.tutor
                        feedback = tutor.get_response(
                            f"Evaluate this answer and provide feedback: {your_answer}",
                            mode="explain",
                            difficulty="medium"
                        )
                        st.success("Answer submitted!")
                        st.markdown("### 📊 Feedback:")
                        st.write(feedback["answer"])
                else:
                    st.warning("Please write an answer first")
        
        with col_b:
            if st.button("Show Hints"):
                if "last_quiz" in st.session_state:
                    with st.spinner("Generating hints..."):
                        tutor = st.session_state.tutor
                        hints = tutor.get_response(
                            "Give hints for answering the quiz questions without revealing the full answers",
                            mode="simplify",
                            difficulty="beginner"
                        )
                        st.markdown("### 💡 Hints:")
                        st.write(hints["answer"])

# Footer with enhanced info
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🎓 AI Tutor - RAG-Powered Learning")
with col2:
    if "content_info" in st.session_state:
        st.caption(f"📚 {len(st.session_state.all_loaded_content)} sources loaded")
with col3:
    if "messages" in st.session_state:
        st.caption(f"💬 {len(st.session_state.messages)} messages")