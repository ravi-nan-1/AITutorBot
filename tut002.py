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
import json

# Configuration
CACHE_DIR = os.path.join(tempfile.gettempdir(), "tutor_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cpu")

# ========== LOGGER CLASS ==========
class TutorLogger:
    """Logger for displaying real-time progress in plain English"""
    
    def __init__(self):
        if 'logs' not in st.session_state:
            st.session_state.logs = []
    
    def log(self, message: str, level: str = "info"):
        """Add a log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "level": level
        }
        st.session_state.logs.append(log_entry)
        return log_entry
    
    def clear_logs(self):
        """Clear all logs"""
        st.session_state.logs = []
    
    def get_logs(self, limit: int = None):
        """Get recent logs"""
        logs = st.session_state.get('logs', [])
        if limit:
            return logs[-limit:]
        return logs

# ========== CONTENT MANAGER ==========
class ContentManager:
    """Manage and display stored content"""
    
    def __init__(self):
        if 'content_database' not in st.session_state:
            st.session_state.content_database = {
                'documents': [],
                'statistics': {
                    'total_documents': 0,
                    'total_chunks': 0,
                    'total_words': 0,
                    'sources': {}
                },
                'all_text': ""  # Store all text for searching
            }
    
    def add_document(self, doc_data: dict):
        """Add a document to the database"""
        doc_id = hashlib.md5(f"{doc_data.get('title', '')}_{datetime.now()}".encode()).hexdigest()[:8]
        doc_data['id'] = doc_id
        doc_data['added_at'] = datetime.now().isoformat()
        
        st.session_state.content_database['documents'].append(doc_data)
        
        # Add to all_text for searching
        if 'content' in doc_data:
            st.session_state.content_database['all_text'] += " " + doc_data['content'].lower()
        
        self.update_statistics()
        return doc_id
    
    def search_content(self, query: str) -> bool:
        """Search if query terms exist in stored content"""
        all_text = st.session_state.content_database.get('all_text', '').lower()
        query_terms = query.lower().split()
        
        # Check if any significant word from query exists in content
        significant_words = [word for word in query_terms if len(word) > 3]
        if significant_words:
            return any(word in all_text for word in significant_words)
        return False
    
    def update_statistics(self):
        """Update content statistics"""
        stats = st.session_state.content_database['statistics']
        docs = st.session_state.content_database['documents']
        
        stats['total_documents'] = len(docs)
        stats['total_chunks'] = sum(doc.get('chunk_count', 0) for doc in docs)
        stats['total_words'] = sum(doc.get('word_count', 0) for doc in docs)
        
        # Count by source type
        source_counts = {}
        for doc in docs:
            source_type = doc.get('source_type', 'unknown')
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        stats['sources'] = source_counts
    
    def get_all_documents(self):
        """Get all stored documents"""
        return st.session_state.content_database['documents']
    
    def get_statistics(self):
        """Get content statistics"""
        return st.session_state.content_database['statistics']
    
    def remove_document(self, doc_id: str):
        """Remove a document by ID"""
        docs = st.session_state.content_database['documents']
        st.session_state.content_database['documents'] = [
            doc for doc in docs if doc.get('id') != doc_id
        ]
        self.update_statistics()
    
    def clear_all(self):
        """Clear all content"""
        st.session_state.content_database = {
            'documents': [],
            'statistics': {
                'total_documents': 0,
                'total_chunks': 0,
                'total_words': 0,
                'sources': {}
            },
            'all_text': ""
        }

# ========== WEB CRAWLER WITH PROGRESS BAR ==========
class TutorWebCrawler:
    def __init__(self, max_pages=15, logger=None):
        self.max_pages = max_pages
        self.visited_urls = set()
        self.scraped_data = []
        self.logger = logger or TutorLogger()
        
    def extract_content(self, url):
        """Extract educational content from URL with logging"""
        try:
            self.logger.log(f"🔍 Connecting to {url[:50]}...", "info")
            response = requests.get(url, timeout=10, verify=False,
                                  headers={'User-Agent': 'Mozilla/5.0'})
            
            if len(response.content) > 2097152:  # 2MB limit
                self.logger.log(f"⚠️ Page too large (>2MB), skipping: {url[:50]}...", "warning")
                return None
            
            self.logger.log(f"📄 Parsing content from {url[:50]}...", "info")
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove non-content elements
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            title = soup.find('title')
            title_text = title.get_text() if title else "No Title"
            
            # Extract main content
            main_content = soup.get_text(separator='\n', strip=True)
            word_count = len(main_content.split())
            
            # Clean and limit text
            main_content = ' '.join(main_content.split())[:15000]
            
            self.logger.log(f"✅ Successfully extracted {word_count} words from: {title_text[:50]}", "success")
            
            return {
                'url': url,
                'title': title_text[:200],
                'content': main_content,
                'word_count': word_count,
                'timestamp': datetime.now().isoformat(),
                'source_type': 'web'
            }
            
        except Exception as e:
            self.logger.log(f"❌ Error crawling {url[:50]}: {str(e)[:100]}", "error")
            return None
    
    def crawl_website(self, start_url):
        """Crawl website for educational content with detailed logging and progress"""
        self.logger.log(f"🚀 Starting web crawl from: {start_url[:50]}...", "info")
        urls_to_visit = [start_url]
        base_domain = urlparse(start_url).netloc
        self.logger.log(f"🌐 Base domain identified: {base_domain}", "info")
        
        # Create progress bar placeholder
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        pages_crawled = 0
        total_words = 0
        
        while urls_to_visit and len(self.scraped_data) < self.max_pages:
            url = urls_to_visit.pop(0)
            
            if url in self.visited_urls:
                self.logger.log(f"⏭️ Skipping already visited: {url[:50]}...", "info")
                continue
            
            self.visited_urls.add(url)
            
            # Update progress
            progress = (pages_crawled + 1) / self.max_pages
            progress_bar.progress(progress)
            status_text.text(f"📖 Loading page {pages_crawled + 1} of {self.max_pages}: {url[:50]}...")
            
            self.logger.log(f"📖 Processing page {pages_crawled + 1}/{self.max_pages}: {url[:50]}...", "info")
            
            data = self.extract_content(url)
            
            if data and len(data['content']) > 100:
                self.scraped_data.append(data)
                pages_crawled += 1
                total_words += data.get('word_count', 0)
                self.logger.log(f"📚 Added page to collection. Total content: {total_words} words", "success")
                
                # Extract links for deeper crawling
                try:
                    self.logger.log(f"🔗 Looking for related links on the page...", "info")
                    response = requests.get(url, timeout=5)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    links_found = 0
                    
                    for link in soup.find_all('a', href=True)[:10]:  # Limit links
                        next_url = urljoin(url, link['href'])
                        if base_domain in next_url and next_url not in self.visited_urls:
                            urls_to_visit.append(next_url)
                            links_found += 1
                    
                    if links_found > 0:
                        self.logger.log(f"🔗 Found {links_found} related links to explore", "info")
                except:
                    pass
            
            time.sleep(0.3)  # Be polite to servers
        
        # Complete progress
        progress_bar.progress(1.0)
        status_text.text(f"✅ Completed! Loaded {pages_crawled} pages with {total_words} words")
        
        self.logger.log(f"✅ Web crawl complete! Collected {pages_crawled} pages with {total_words} total words", "success")
        return self.scraped_data

# ========== CONTENT PROCESSORS WITH LOGGING ==========
class ContentProcessor:
    def __init__(self, logger=None):
        self.logger = logger or TutorLogger()
    
    def process_pdf(self, pdf_file):
        """Extract text from PDF file with logging"""
        try:
            self.logger.log(f"📄 Opening PDF file: {pdf_file.name}", "info")
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
            total_pages = len(pdf_reader.pages)
            self.logger.log(f"📖 PDF contains {total_pages} pages", "info")
            
            text = ""
            pages_processed = 0
            
            for page_num in range(min(total_pages, 50)):  # Limit to 50 pages
                progress = (page_num + 1) / min(total_pages, 50)
                progress_bar.progress(progress)
                status_text.text(f"📃 Reading page {page_num + 1} of {min(total_pages, 50)}...")
                
                self.logger.log(f"📃 Reading page {page_num + 1}/{min(total_pages, 50)}...", "info")
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n"
                pages_processed += 1
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Completed! Processed {pages_processed} pages")
            
            word_count = len(text.split())
            self.logger.log(f"✅ Successfully extracted {word_count} words from {pages_processed} pages", "success")
            
            return {
                'title': pdf_file.name,
                'content': text,
                'source_type': 'pdf',
                'word_count': word_count,
                'page_count': pages_processed,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.log(f"❌ Error processing PDF: {str(e)[:100]}", "error")
            return None
    
    def process_youtube(self, url):
        """Extract transcript from YouTube video with logging"""
        try:
            self.logger.log(f"🎥 Processing YouTube video: {url[:50]}...", "info")
            
            # Create progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Extract video ID
            video_id = None
            if "youtube.com/watch?v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[1].split("?")[0]
            
            if not video_id:
                self.logger.log("❌ Could not extract video ID from URL", "error")
                return None
            
            self.logger.log(f"🎬 Video ID identified: {video_id}", "info")
            
            progress_bar.progress(0.3)
            status_text.text("🔍 Looking for video transcripts...")
            
            # Get transcript
            self.logger.log("📝 Fetching available transcripts...", "info")
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            progress_bar.progress(0.6)
            status_text.text("📝 Processing transcript...")
            
            # Try to get English transcript first
            transcript = None
            available_languages = []
            
            for t in transcript_list:
                available_languages.append(t.language)
                if t.language_code.startswith('en'):
                    transcript = t.fetch()
                    self.logger.log(f"✅ Found English transcript", "success")
                    break
            
            # If no English, get any available
            if not transcript:
                self.logger.log(f"⚠️ No English transcript found. Available languages: {', '.join(available_languages[:5])}", "warning")
                transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB']).fetch()
            
            progress_bar.progress(0.9)
            status_text.text("📄 Finalizing transcript...")
            
            # Combine transcript text
            self.logger.log("📄 Processing transcript text...", "info")
            text = " ".join([entry['text'] for entry in transcript])
            word_count = len(text.split())
            
            # Calculate video duration
            if transcript:
                duration = transcript[-1]['start'] if transcript else 0
                duration_min = int(duration / 60)
                self.logger.log(f"⏱️ Video duration: approximately {duration_min} minutes", "info")
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Completed! Extracted {word_count} words")
            
            self.logger.log(f"✅ Successfully extracted {word_count} words from video transcript", "success")
            
            return {
                'title': f"YouTube Video: {video_id}",
                'content': text,
                'source_type': 'youtube',
                'url': url,
                'word_count': word_count,
                'video_id': video_id,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.log(f"❌ Error processing YouTube video: {str(e)[:100]}", "error")
            return None
    
    def process_audio(self, audio_file):
        """Extract text from audio file using speech recognition with logging"""
        try:
            self.logger.log(f"🎵 Processing audio file: {audio_file.name}", "info")
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            file_size_mb = len(audio_file.getvalue()) / (1024 * 1024)
            self.logger.log(f"📊 File size: {file_size_mb:.2f} MB", "info")
            
            # Save temporary file
            temp_path = os.path.join(CACHE_DIR, f"temp_audio_{hashlib.md5(audio_file.name.encode()).hexdigest()}")
            
            progress_bar.progress(0.2)
            status_text.text("💾 Saving temporary file...")
            self.logger.log("💾 Saving temporary file...", "info")
            
            with open(temp_path, "wb") as f:
                f.write(audio_file.read())
            
            # Convert to wav if needed
            progress_bar.progress(0.4)
            status_text.text("🔄 Converting audio format...")
            self.logger.log("🔄 Converting audio format...", "info")
            
            audio = AudioSegment.from_file(temp_path)
            duration_seconds = len(audio) / 1000
            self.logger.log(f"⏱️ Audio duration: {duration_seconds:.1f} seconds", "info")
            
            wav_path = temp_path + ".wav"
            audio.export(wav_path, format="wav")
            
            # Recognize speech
            progress_bar.progress(0.7)
            status_text.text("🎤 Starting speech recognition...")
            self.logger.log("🎤 Starting speech recognition...", "info")
            
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                
                progress_bar.progress(0.9)
                status_text.text("🧠 Processing speech to text...")
                self.logger.log("🧠 Processing speech to text...", "info")
                
                text = recognizer.recognize_google(audio_data)
            
            word_count = len(text.split())
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Completed! Transcribed {word_count} words")
            self.logger.log(f"✅ Successfully transcribed {word_count} words from audio", "success")
            
            # Clean up
            os.remove(temp_path)
            os.remove(wav_path)
            self.logger.log("🧹 Cleaned up temporary files", "info")
            
            return {
                'title': audio_file.name,
                'content': text,
                'source_type': 'audio',
                'word_count': word_count,
                'duration': duration_seconds,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.log(f"❌ Error processing audio: {str(e)[:100]}", "error")
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
            prompt = prompt.replace('\n\n', '\n').strip()
            
            if "Context:" in prompt or "Based on:" in prompt or "Using:" in prompt:
                prompt = f"Answer using only the given information. {prompt}"
            
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
            
            response = response.strip()
            if not response or len(response) < 20:
                response = "Based on the provided context, I need more specific information to answer your question properly. Please provide more details or rephrase your question."
            
            gc.collect()
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an issue. Please try rephrasing your question or provide more context."

# ========== ENHANCED RAG TUTOR WITH CONTENT CHECKING ==========
class RAGTutor:
    def __init__(self, vectorstore, llm, content_manager=None):
        self.vectorstore = vectorstore
        self.llm = llm
        self.conversation_history = []
        self.all_content = []
        self.content_manager = content_manager or ContentManager()
    
    def set_all_content(self, content):
        self.all_content = content
    
    def create_prompt(self, question: str, context: str, mode: str, difficulty: str) -> str:
        context = context[:1500]
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
        try:
            # First check if the question content exists in our database
            content_exists = self.content_manager.search_content(question)
            
            # Search for relevant documents
            docs = self.vectorstore.similarity_search(question, k=6)
            
            # If no relevant docs found and content doesn't exist
            if (not docs or len(docs) == 0) and not content_exists:
                return {
                    "answer": f"❌ **Content not found.** The topic '{question[:50]}' is not available in the loaded content. Please make sure you've loaded relevant material that covers this topic.",
                    "sources": [],
                    "suggestions": [
                        "Load more content",
                        "Try a different topic", 
                        "Check loaded content in Content Library",
                        "Rephrase your question"
                    ]
                }
            
            # Check relevance of found documents
            relevant_docs = []
            for doc in docs:
                doc_text = doc.page_content.lower()
                query_words = question.lower().split()
                # Check if at least some query words appear in the document
                if any(word in doc_text for word in query_words if len(word) > 3):
                    relevant_docs.append(doc)
            
            # If no relevant documents found
            if not relevant_docs:
                return {
                    "answer": f"❌ **Topic not found in loaded content.** I couldn't find information about '{question[:50]}' in the current materials. The loaded content may not cover this specific topic.",
                    "sources": [],
                    "suggestions": [
                        "Browse Content Library to see what's available",
                        "Load content related to this topic",
                        "Ask about topics from the loaded materials",
                        "Try broader terms"
                    ]
                }
            
            # Use relevant documents for response
            context_parts = []
            for doc in relevant_docs[:6]:
                content = doc.page_content[:400]
                context_parts.append(content)
            
            context = " ".join(context_parts)
            
            # Generate response
            prompt = self.create_prompt(question, context, mode.lower(), difficulty.lower())
            response = self.llm._call(prompt)
            
            if len(response) < 30:
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
            
            # Extract sources
            sources = []
            seen_titles = set()
            for doc in relevant_docs[:4]:
                title = doc.metadata.get("title", "Unknown")
                if title not in seen_titles:
                    sources.append({
                        "title": title,
                        "url": doc.metadata.get("url", "#"),
                        "type": doc.metadata.get("source_type", "web"),
                        "snippet": doc.page_content[:100] + "..."
                    })
                    seen_titles.add(title)
            
            # Generate dynamic suggestions
            suggestions = self.generate_dynamic_suggestions(question, relevant_docs)
            
            return {
                "answer": response,
                "sources": sources,
                "suggestions": suggestions,
                "context_used": context[:200] + "..."
            }
            
        except Exception as e:
            return {
                "answer": f"❌ Error processing your question. Please try again with different wording.",
                "sources": [],
                "suggestions": ["Try a simpler question", "Check Content Library", "Load more content"]
            }
    
    def generate_dynamic_suggestions(self, question: str, docs: List) -> List[str]:
        suggestions = []
        
        if docs:
            all_text = " ".join([doc.page_content[:200] for doc in docs])
            words = re.findall(r'\b[A-Z][a-z]+\b', all_text)
            unique_topics = list(set(words))[:3]
            
            if unique_topics:
                suggestions.append(f"Tell me more about {unique_topics[0]}")
                if len(unique_topics) > 1:
                    suggestions.append(f"Explain {unique_topics[1]}")
            
            suggestions.append("Give me an example")
            suggestions.append("Quiz me on this topic")
        else:
            suggestions = [
                "Load more content first",
                "Try a different topic",
                "Ask a general question"
            ]
        
        return suggestions[:4]

# ========== HELPER FUNCTION ==========
def process_loaded_content(content_data, logger, content_manager):
    """Process and vectorize loaded content with logging"""
    try:
        logger.log("🔧 Starting content processing...", "info")
        
        # Create embeddings
        logger.log("🧠 Initializing embedding model...", "info")
        embeddings = TutorEmbeddings()
        
        # Process text
        texts = []
        metadata = []
        
        logger.log("✂️ Splitting content into chunks...", "info")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )
        
        total_chunks = 0
        for item in content_data:
            chunks = splitter.split_text(item['content'])
            logger.log(f"📄 Processing {len(chunks)} chunks from {item.get('title', 'Unknown')[:30]}...", "info")
            
            for i, chunk in enumerate(chunks[:20]):
                texts.append(chunk)
                metadata.append({
                    'url': item.get('url', ''),
                    'title': item.get('title', 'Unknown'),
                    'source_type': item.get('source_type', 'unknown'),
                    'chunk_index': i
                })
                total_chunks += 1
            
            # Update content manager with chunk count
            item['chunk_count'] = len(chunks[:20])
        
        logger.log(f"✅ Created {total_chunks} text chunks", "success")
        
        # Create or update vector store
        logger.log("🔗 Creating vector embeddings...", "info")
        if "vectorstore" in st.session_state:
            new_vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
            st.session_state.vectorstore.merge_from(new_vectorstore)
            logger.log("✅ Updated existing vector store", "success")
        else:
            st.session_state.vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
            logger.log("✅ Created new vector store", "success")
        
        # Initialize LLM and Tutor
        if "llm" not in st.session_state:
            logger.log("🤖 Initializing language model...", "info")
            st.session_state.llm = TutorLLM()
            logger.log("✅ Language model ready", "success")
        
        logger.log("🎓 Initializing tutor system...", "info")
        st.session_state.tutor = RAGTutor(st.session_state.vectorstore, st.session_state.llm, content_manager)
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
        
        logger.log(f"🎉 Successfully processed all content! Ready for Q&A", "success")
        st.success(f"✅ Successfully loaded {len(texts)} text chunks!")
        st.info("Go to 'Chat & Learn' tab to start learning!")
        
    except Exception as e:
        logger.log(f"❌ Error processing content: {str(e)[:200]}", "error")
        st.error(f"Error processing content: {e}")

# ========== STREAMLIT UI WITH ENHANCED FEATURES ==========
st.set_page_config(page_title="AI Tutor Assistant", layout="wide", page_icon="🎓")

# Custom CSS
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
    .log-entry {
        padding: 0.3rem;
        margin: 0.2rem 0;
        border-left: 3px solid #2196F3;
        background-color: #f5f5f5;
    }
    .log-success {
        border-left-color: #4CAF50;
        background-color: #e8f5e9;
    }
    .log-warning {
        border-left-color: #FF9800;
        background-color: #fff3e0;
    }
    .log-error {
        border-left-color: #f44336;
        background-color: #ffebee;
    }
    .content-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: white;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stChatMessage):last-of-type {
        padding-bottom: 5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
logger = TutorLogger()
content_manager = ContentManager()

st.title("🎓 AI Tutor - Your Learning Assistant")

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
    
    # Content Statistics
    stats = content_manager.get_statistics()
    if stats['total_documents'] > 0:
        st.markdown("### 📊 Content Statistics")
        st.metric("Total Documents", stats['total_documents'])
        st.metric("Total Chunks", stats['total_chunks'])
        st.metric("Total Words", f"{stats['total_words']:,}")
        
        if stats['sources']:
            st.markdown("**Sources:**")
            for source_type, count in stats['sources'].items():
                st.caption(f"• {source_type}: {count}")
    
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
            content_manager.clear_all()
            logger.clear_logs()
            st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat & Learn", "📚 Load Content", "📝 Practice", "🗂️ Content Library", "📋 Activity Logs"])

# ========== TAB 2: LOAD CONTENT WITH WORKING PROGRESS BAR ==========
with tab2:
    st.header("📚 Load Learning Content")
    
    # Content type selector
    content_type = st.radio(
        "Select content type:",
        ["🌐 Website", "📄 PDF", "🎥 YouTube", "🎵 Audio"],
        horizontal=True
    )
    
    # Create columns for main content and logs
    col_main, col_logs = st.columns([2, 1])
    
    with col_main:
        content_processor = ContentProcessor(logger=logger)
        
        if content_type == "🌐 Website":
            url_input = st.text_input(
                "Enter URL to learn from:",
                placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence"
            )
            
            # Enhanced quick examples
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
                    logger.clear_logs()
                    with st.spinner("Loading website content..."):
                        crawler = TutorWebCrawler(max_pages=max_pages, logger=logger)
                        scraped_data = crawler.crawl_website(url_input)
                        
                        if scraped_data:
                            # Add to content manager
                            for doc in scraped_data:
                                content_manager.add_document(doc)
                            
                            st.session_state.all_loaded_content.extend(scraped_data)
                            process_loaded_content(scraped_data, logger, content_manager)
        
        elif content_type == "📄 PDF":
            pdf_file = st.file_uploader("Upload PDF file", type=['pdf'])
            
            if pdf_file and st.button("📄 Load PDF", type="primary"):
                logger.clear_logs()
                with st.spinner("Processing PDF..."):
                    pdf_data = content_processor.process_pdf(pdf_file)
                    if pdf_data:
                        content_manager.add_document(pdf_data)
                        st.session_state.all_loaded_content.append(pdf_data)
                        process_loaded_content([pdf_data], logger, content_manager)
        
        elif content_type == "🎥 YouTube":
            youtube_url = st.text_input(
                "Enter YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=..."
            )
            
            st.markdown("**Popular Educational Channels:**")
            examples = ["Khan Academy", "MIT OpenCourseWare", "CrashCourse", "TED-Ed"]
            st.caption(" • ".join(examples))
            
            if youtube_url and st.button("🎥 Load Video", type="primary"):
                logger.clear_logs()
                with st.spinner("Extracting video transcript..."):
                    video_data = content_processor.process_youtube(youtube_url)
                    if video_data:
                        content_manager.add_document(video_data)
                        st.session_state.all_loaded_content.append(video_data)
                        process_loaded_content([video_data], logger, content_manager)
        
        elif content_type == "🎵 Audio":
            audio_file = st.file_uploader(
                "Upload audio file", 
                type=['mp3', 'wav', 'ogg', 'm4a']
            )
            
            if audio_file and st.button("🎵 Process Audio", type="primary"):
                logger.clear_logs()
                with st.spinner("Transcribing audio..."):
                    audio_data = content_processor.process_audio(audio_file)
                    if audio_data:
                        content_manager.add_document(audio_data)
                        st.session_state.all_loaded_content.append(audio_data)
                        process_loaded_content([audio_data], logger, content_manager)
    
    # Live logs display
    with col_logs:
        st.markdown("### 📋 Loading Progress")
        log_container = st.container()
        
        with log_container:
            logs = logger.get_logs(limit=20)  # Show last 20 logs
            for log in logs:
                level_class = f"log-{log['level']}"
                icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(log['level'], "📝")
                st.markdown(
                    f'<div class="{level_class}" style="padding: 0.3rem; margin: 0.2rem 0; border-radius: 4px;">'
                    f'{icon} <small>{log["timestamp"]}</small><br>'
                    f'{log["message"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ========== TAB 1: CHAT INTERFACE WITH CONTENT CHECKING ==========
with tab1:
    if "content_loaded" not in st.session_state:
        st.info("👈 Please load content from the 'Load Content' tab first")
        
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

# ========== TAB 3: PRACTICE ==========
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

# ========== TAB 4: CONTENT LIBRARY ==========
with tab4:
    st.header("🗂️ Content Library")
    st.markdown("View and manage all your loaded content")
    
    # Filter options
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        filter_type = st.selectbox(
            "Filter by type:",
            ["All", "web", "pdf", "youtube", "audio"]
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            ["Newest", "Oldest", "Largest", "Title"]
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Display documents
    documents = content_manager.get_all_documents()
    
    if documents:
        # Apply filters
        if filter_type != "All":
            documents = [doc for doc in documents if doc.get('source_type') == filter_type]
        
        # Apply sorting
        if sort_by == "Newest":
            documents.sort(key=lambda x: x.get('added_at', ''), reverse=True)
        elif sort_by == "Oldest":
            documents.sort(key=lambda x: x.get('added_at', ''))
        elif sort_by == "Largest":
            documents.sort(key=lambda x: x.get('word_count', 0), reverse=True)
        elif sort_by == "Title":
            documents.sort(key=lambda x: x.get('title', ''))
        
        # Display each document
        for doc in documents:
            with st.expander(f"{doc.get('source_type', 'unknown').upper()} - {doc.get('title', 'Unknown')[:50]}..."):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**ID:** {doc.get('id', 'N/A')}")
                    st.markdown(f"**Type:** {doc.get('source_type', 'unknown')}")
                    st.markdown(f"**Words:** {doc.get('word_count', 0):,}")
                    st.markdown(f"**Added:** {doc.get('added_at', 'Unknown')[:19]}")
                    
                    if doc.get('url'):
                        st.markdown(f"**URL:** {doc.get('url', '')[:50]}...")
                    
                    # Show preview
                    if doc.get('content'):
                        st.markdown("**Preview:**")
                        st.text(doc['content'][:500] + "...")
                
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button(f"🗑️ Remove", key=f"remove_{doc.get('id', '')}"):
                        content_manager.remove_document(doc.get('id'))
                        st.success(f"Removed: {doc.get('title', 'Unknown')[:30]}...")
                        st.rerun()
    else:
        st.info("📚 No content loaded yet. Go to 'Load Content' tab to add learning materials.")
    
    # Summary statistics
    st.markdown("---")
    stats = content_manager.get_statistics()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Documents", stats['total_documents'])
    with col2:
        st.metric("📦 Total Chunks", stats['total_chunks'])
    with col3:
        st.metric("📝 Total Words", f"{stats['total_words']:,}")
    with col4:
        st.metric("🌐 Source Types", len(stats['sources']))

# ========== TAB 5: ACTIVITY LOGS ==========
with tab5:
    st.header("📋 Activity Logs")
    st.markdown("View all system activities and operations")
    
    # Log controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        log_limit = st.selectbox("Show last:", [20, 50, 100, "All"])
    with col2:
        log_level_filter = st.multiselect(
            "Filter by level:",
            ["info", "success", "warning", "error"],
            default=["info", "success", "warning", "error"]
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Logs"):
            logger.clear_logs()
            st.success("Logs cleared!")
            st.rerun()
    
    # Display logs
    all_logs = logger.get_logs()
    
    if all_logs:
        # Apply filters
        filtered_logs = [log for log in all_logs if log['level'] in log_level_filter]
        
        # Apply limit
        if log_limit != "All":
            filtered_logs = filtered_logs[-log_limit:]
        
        # Reverse to show newest first
        filtered_logs.reverse()
        
        # Display logs in a scrollable container
        log_container = st.container()
        with log_container:
            for log in filtered_logs:
                level_style = {
                    "info": "border-left: 3px solid #2196F3; background-color: #e3f2fd;",
                    "success": "border-left: 3px solid #4CAF50; background-color: #e8f5e9;",
                    "warning": "border-left: 3px solid #FF9800; background-color: #fff3e0;",
                    "error": "border-left: 3px solid #f44336; background-color: #ffebee;"
                }.get(log['level'], "")
                
                icon = {
                    "info": "ℹ️",
                    "success": "✅",
                    "warning": "⚠️",
                    "error": "❌"
                }.get(log['level'], "📝")
                
                st.markdown(
                    f'<div style="{level_style} padding: 0.5rem; margin: 0.3rem 0; border-radius: 4px;">'
                    f'<strong>{icon} {log["timestamp"]}</strong><br>'
                    f'{log["message"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("No logs available. Activities will appear here when you load content or interact with the system.")
    
    # Export logs option
    if all_logs:
        st.markdown("---")
        if st.button("📥 Export Logs as JSON"):
            logs_json = json.dumps(all_logs, indent=2)
            st.download_button(
                label="Download Logs",
                data=logs_json,
                file_name=f"tutor_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Footer with enhanced info
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.caption("🎓 AI Tutor - RAG Learning")
with col2:
    if "content_info" in st.session_state:
        st.caption(f"📚 {len(st.session_state.all_loaded_content)} sources")
with col3:
    if "messages" in st.session_state:
        st.caption(f"💬 {len(st.session_state.messages)} messages")
with col4:
    logs = logger.get_logs()
    if logs:
        st.caption(f"📋 {len(logs)} activities logged")