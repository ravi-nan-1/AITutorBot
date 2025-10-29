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
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import speech_recognition as sr
from pydub import AudioSegment
import hashlib
import json
import pickle
from pathlib import Path
import shutil
import nltk
from collections import Counter
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# ========== PERSISTENCE CONFIGURATION ==========
PERSISTENCE_DIR = os.path.join(tempfile.gettempdir(), "tutor_persistence")
os.makedirs(PERSISTENCE_DIR, exist_ok=True)
CACHE_DIR = os.path.join(tempfile.gettempdir(), "tutor_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cpu")


def save_session_state():
    """Save critical session state to disk"""
    try:
        persistence_data = {}
        
        # Save content database
        if 'content_database' in st.session_state:
            persistence_data['content_database'] = st.session_state.content_database
        
        # Save loaded content
        if 'all_loaded_content' in st.session_state:
            persistence_data['all_loaded_content'] = st.session_state.all_loaded_content
        
        # Save conversation history
        if 'messages' in st.session_state:
            persistence_data['messages'] = st.session_state.messages
        
        # Save content info
        if 'content_info' in st.session_state:
            persistence_data['content_info'] = st.session_state.content_info
        
        # Save content_loaded flag
        if 'content_loaded' in st.session_state:
            persistence_data['content_loaded'] = st.session_state.content_loaded
        
        # Save summaries database
        if 'summaries_database' in st.session_state:
            persistence_data['summaries_database'] = st.session_state.summaries_database
        
        # Save to file
        persistence_file = os.path.join(PERSISTENCE_DIR, "tutor_data.pkl")
        with open(persistence_file, 'wb') as f:
            pickle.dump(persistence_data, f)
            
        return True
    except Exception as e:
        st.error(f"Error saving session: {e}")
        return False

def load_session_state():
    """Load session state from disk"""
    try:
        persistence_file = os.path.join(PERSISTENCE_DIR, "tutor_data.pkl")
        if os.path.exists(persistence_file):
            with open(persistence_file, 'rb') as f:
                persistence_data = pickle.load(f)
            
            # Restore session state
            for key, value in persistence_data.items():
                st.session_state[key] = value
                
            return True
    except Exception as e:
        st.error(f"Error loading session: {e}")
    return False

def clear_persistence_files():
    """Clear all persistence files"""
    try:
        persistence_files = [
            os.path.join(PERSISTENCE_DIR, "tutor_data.pkl"),
            os.path.join(PERSISTENCE_DIR, "content_database.json"),
            os.path.join(PERSISTENCE_DIR, "vectorstore")
        ]
        
        for file_path in persistence_files:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
        return True
    except Exception as e:
        st.error(f"Error clearing persistence: {e}")
        return False

# [Keep all the existing persistence functions, ContentSummarizer, SummaryManager, TutorLogger, ContentManager, TutorWebCrawler classes as they are]
# ... (all the existing code for these classes remains the same)

# ========== FIXED CONTENT PROCESSOR CLASS ==========
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
        """FIXED: Extract transcript from YouTube video with logging"""
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
            elif "youtube.com/embed/" in url:
                video_id = url.split("embed/")[1].split("?")[0]
            
            if not video_id:
                self.logger.log("❌ Could not extract video ID from URL", "error")
                progress_bar.empty()
                status_text.empty()
                return None
            
            self.logger.log(f"🎬 Video ID identified: {video_id}", "info")
            
            progress_bar.progress(0.3)
            status_text.text("🔍 Looking for video transcripts...")
            
            # FIXED: Get transcript using correct method
            self.logger.log("📝 Fetching available transcripts...", "info")
            
            transcript = None
            transcript_text = ""
            
            try:
                # Try to get transcript with language preference
                preferred_languages = ['en', 'en-US', 'en-GB', 'en-AU', 'en-CA']
                
                for lang in preferred_languages:
                    try:
                        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                        self.logger.log(f"✅ Found transcript in language: {lang}", "success")
                        break
                    except:
                        continue
                
                # If no English transcript found, try to get any available transcript
                if not transcript:
                    self.logger.log("⚠️ No English transcript found, trying auto-generated or other languages...", "warning")
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    self.logger.log("✅ Found alternative transcript", "success")
                    
            except TranscriptsDisabled:
                self.logger.log("❌ Transcripts are disabled for this video", "error")
                progress_bar.empty()
                status_text.empty()
                return None
            except NoTranscriptFound:
                self.logger.log("❌ No transcripts found for this video", "error")
                progress_bar.empty()
                status_text.empty()
                return None
            except VideoUnavailable:
                self.logger.log("❌ Video is unavailable or doesn't exist", "error")
                progress_bar.empty()
                status_text.empty()
                return None
            except Exception as e:
                self.logger.log(f"❌ Error fetching transcript: {str(e)[:100]}", "error")
                progress_bar.empty()
                status_text.empty()
                return None
            
            if not transcript:
                self.logger.log("❌ No transcript available for this video", "error")
                progress_bar.empty()
                status_text.empty()
                return None
            
            progress_bar.progress(0.6)
            status_text.text("📝 Processing transcript...")
            
            progress_bar.progress(0.9)
            status_text.text("📄 Finalizing transcript...")
            
            # Combine transcript text
            self.logger.log("📄 Processing transcript text...", "info")
            text = " ".join([entry['text'] for entry in transcript])
            word_count = len(text.split())
            
            # Calculate video duration
            duration_seconds = 0
            if transcript and len(transcript) > 0:
                last_entry = transcript[-1]
                duration_seconds = last_entry.get('start', 0) + last_entry.get('duration', 0)
                duration_min = int(duration_seconds / 60)
                self.logger.log(f"⏱️ Video duration: approximately {duration_min} minutes", "info")
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Completed! Extracted {word_count} words")
            
            # Clean up progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            self.logger.log(f"✅ Successfully extracted {word_count} words from video transcript", "success")
            
            return {
                'title': f"YouTube Video: {video_id}",
                'content': text,
                'source_type': 'youtube',
                'url': url,
                'word_count': word_count,
                'video_id': video_id,
                'duration_seconds': duration_seconds,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"❌ Error processing YouTube video: {str(e)[:100]}", "error")
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()
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
    
    def process_text(self, text_input):
        """Process plain text input"""
        try:
            self.logger.log("📝 Processing text input...", "info")
            
            word_count = len(text_input.split())
            
            self.logger.log(f"✅ Successfully processed {word_count} words from text input", "success")
            
            return {
                'title': f"Text Input - {text_input[:50]}...",
                'content': text_input,
                'source_type': 'text',
                'word_count': word_count,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.log(f"❌ Error processing text: {str(e)[:100]}", "error")
            return None
    
    def process_web_url(self, url):
        """Process web URL for content extraction"""
        try:
            self.logger.log(f"🌐 Processing web URL: {url[:50]}...", "info")
            
            # Use the web crawler for single page
            crawler = TutorWebCrawler(max_pages=1, logger=self.logger)
            scraped_data = crawler.crawl_website(url)
            
            if scraped_data and len(scraped_data) > 0:
                return scraped_data[0]
            
            return None
        except Exception as e:
            self.logger.log(f"❌ Error processing web URL: {str(e)[:100]}", "error")
            return None

# [Keep all other existing classes as they are - TutorEmbeddings, TutorLLM, RAGTutor, etc.]

# ========== STREAMLIT UI WITH ENHANCED SUMMARIZATION ==========
st.set_page_config(page_title="AI Tutor Assistant", layout="wide", page_icon="🎓")

# Load previous session state on startup
if not load_session_state():
    st.sidebar.info("🔍 Starting fresh session...")
else:
    st.sidebar.success("♻️ Previous session restored!")

# [Keep all the existing CSS and initialization code]

# Initialize components
logger = TutorLogger()
content_manager = ContentManager()
summary_manager = SummaryManager()

st.title("🎓 AI Tutor - Your Learning Assistant")

# Initialize session state
if "all_loaded_content" not in st.session_state:
    st.session_state.all_loaded_content = []

# [Keep all the sidebar code as is]

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Chat & Learn", "📚 Load Content", "📊 Summarize Content", "📝 Practice", "🗂️ Content Library", "📋 Activity Logs"])

# ========== ENHANCED TAB 3: CONTENT SUMMARIZATION ==========
with tab3:
    st.header("📊 Content Summarization Hub")
    st.markdown("Generate comprehensive summaries with keywords and tags from any content source")
    
    # Create two sub-tabs for existing and external content
    sub_tab1, sub_tab2 = st.tabs(["📚 Existing Content", "🌐 External Content"])
    
    # SUB-TAB 1: EXISTING CONTENT
    with sub_tab1:
        st.subheader("📖 Summarize Loaded Content")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get all available content
            all_documents = content_manager.get_all_documents()
            
            if not all_documents:
                st.info("📚 No content available for summarization. Please load content first in the 'Load Content' tab.")
                st.stop()
            
            # Content selection
            content_options = {f"{doc['title'][:50]}... ({doc['source_type']})": doc for doc in all_documents}
            selected_content_key = st.selectbox(
                "Choose content to summarize:",
                options=list(content_options.keys()),
                help="Select from your loaded content"
            )
            
            selected_content = content_options[selected_content_key]
            
            # Check if already summarized
            all_summaries = summary_manager.get_all_summaries()
            already_summarized = any(
                summary.get('title') == selected_content['title'] 
                for summary in all_summaries
            )
            
            if already_summarized:
                st.warning("⚠️ This content has already been summarized. Generating a new summary will replace the existing one.")
            
            # Display content preview
            with st.expander("📄 Content Preview", expanded=False):
                st.text(selected_content.get('content', '')[:1000] + "...")
                st.caption(f"Total words: {selected_content.get('word_count', 0):,}")
        
        with col2:
            st.markdown("#### 📊 Content Details")
            st.caption(f"**Type:** {selected_content.get('source_type', 'unknown').upper()}")
            st.caption(f"**Words:** {selected_content.get('word_count', 0):,}")
            st.caption(f"**Added:** {selected_content.get('added_at', 'Unknown')[:19]}")
            
            if selected_content.get('url'):
                st.caption(f"**Source:** {selected_content.get('url', '')[:30]}...")
        
        # Generate summary button
        if st.button("🚀 Generate Summary from Existing Content", type="primary", use_container_width=True):
            with st.spinner("Analyzing content and generating comprehensive summary..."):
                summary_id = summary_manager.add_summary(selected_content)
                
                if summary_id:
                    st.success("✅ Summary generated successfully!")
                    
                    # Display the generated summary
                    summary_data = summary_manager.get_summary(summary_id)
                    if summary_data:
                        display_summary_card(summary_data)
                        st.info("💡 This summary is now available in the chat! You can ask questions about it.")
                else:
                    st.error("❌ Failed to generate summary. Please try again.")
    
    # SUB-TAB 2: EXTERNAL CONTENT
    with sub_tab2:
        st.subheader("🌍 Summarize External Content")
        st.markdown("Load and summarize content from external sources directly")
        
        # Content type selector
        external_content_type = st.radio(
            "Select external content type:",
            ["📝 Text", "🌐 Web URL", "📄 PDF", "🎥 YouTube", "🎵 Audio"],
            horizontal=True,
            key="external_content_type"
        )
        
        content_processor = ContentProcessor(logger=logger)
        external_content_data = None
        
        # Different input methods based on content type
        if external_content_type == "📝 Text":
            text_input = st.text_area(
                "Enter or paste your text:",
                height=200,
                placeholder="Paste any text content you want to summarize..."
            )
            
            if text_input and st.button("📝 Process Text", type="primary"):
                with st.spinner("Processing text..."):
                    external_content_data = content_processor.process_text(text_input)
        
        elif external_content_type == "🌐 Web URL":
            web_url = st.text_input(
                "Enter web page URL:",
                placeholder="https://example.com/article"
            )
            
            if web_url and st.button("🌐 Load Web Page", type="primary"):
                with st.spinner("Loading web content..."):
                    external_content_data = content_processor.process_web_url(web_url)
        
        elif external_content_type == "📄 PDF":
            pdf_file = st.file_uploader(
                "Upload PDF file:", 
                type=['pdf'],
                key="external_pdf"
            )
            
            if pdf_file and st.button("📄 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    external_content_data = content_processor.process_pdf(pdf_file)
        
        elif external_content_type == "🎥 YouTube":
            youtube_url = st.text_input(
                "Enter YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=..."
            )
            
            if youtube_url and st.button("🎥 Load YouTube Video", type="primary"):
                with st.spinner("Extracting video transcript..."):
                    external_content_data = content_processor.process_youtube(youtube_url)
        
        elif external_content_type == "🎵 Audio":
            audio_file = st.file_uploader(
                "Upload audio file:",
                type=['mp3', 'wav', 'ogg', 'm4a'],
                key="external_audio"
            )
            
            if audio_file and st.button("🎵 Process Audio", type="primary"):
                with st.spinner("Transcribing audio..."):
                    external_content_data = content_processor.process_audio(audio_file)
        
        # Process and summarize external content
        if external_content_data:
            st.success(f"✅ Successfully loaded {external_content_data.get('word_count', 0):,} words!")
            
            # Show content preview
            with st.expander("📄 Loaded Content Preview", expanded=True):
                st.text(external_content_data.get('content', '')[:1000] + "...")
            
            # Generate summary options
            col1, col2 = st.columns(2)
            with col1:
                save_to_library = st.checkbox("💾 Save to Content Library", value=True)
            with col2:
                make_available_for_chat = st.checkbox("💬 Make available for chat", value=True)
            
            if st.button("🎯 Generate Summary", type="primary", use_container_width=True):
                with st.spinner("Generating comprehensive summary..."):
                    # Save to library if requested
                    if save_to_library:
                        content_manager.add_document(external_content_data)
                        st.session_state.all_loaded_content.append(external_content_data)
                    
                    # Generate summary
                    summary_id = summary_manager.add_summary(external_content_data)
                    
                    if summary_id:
                        st.success("✅ Summary generated successfully!")
                        
                        # Display the generated summary
                        summary_data = summary_manager.get_summary(summary_id)
                        if summary_data:
                            display_summary_card(summary_data)
                            
                            if make_available_for_chat:
                                # Process for chat availability
                                process_loaded_content([external_content_data], logger, content_manager, summary_manager)
                                st.info("💡 Content and summary are now available in the chat!")
                    else:
                        st.error("❌ Failed to generate summary. Please try again.")
    
    # Display all existing summaries
    st.markdown("---")
    st.subheader("📂 Summary Library")
    
    all_summaries = summary_manager.get_all_summaries()
    
    if all_summaries:
        # Search and filter
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_query = st.text_input("🔍 Search summaries:", placeholder="Enter keyword or tag...")
        with col2:
            filter_type = st.selectbox("Filter by type:", ["All"] + list(set(s['source_type'] for s in all_summaries)))
        with col3:
            sort_by = st.selectbox("Sort by:", ["Newest", "Oldest", "Title"])
        
        # Apply filters
        display_summaries = all_summaries
        
        if search_query:
            display_summaries = summary_manager.search_summaries(search_query)
        
        if filter_type != "All":
            display_summaries = [s for s in display_summaries if s['source_type'] == filter_type]
        
        # Apply sorting
        if sort_by == "Newest":
            display_summaries = sorted(display_summaries, key=lambda x: x['timestamp'], reverse=True)
        elif sort_by == "Oldest":
            display_summaries = sorted(display_summaries, key=lambda x: x['timestamp'])
        elif sort_by == "Title":
            display_summaries = sorted(display_summaries, key=lambda x: x['title'])
        
        st.info(f"📊 Showing {len(display_summaries)} of {len(all_summaries)} summaries")
        
        # Display summaries in a grid
        for summary in display_summaries:
            with st.expander(f"📊 {summary['title'][:60]}... ({summary['source_type']})", expanded=False):
                display_summary_card(summary, show_remove=True)
    else:
        st.info("📚 No summaries yet. Generate summaries from existing content or load external content to summarize.")

def display_summary_card(summary_data, show_remove=False):
    """Helper function to display a summary card"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Two-liner
        st.markdown("**📝 Quick Summary:**")
        st.info(summary_data['two_liner'])
        
        # Detailed summary
        st.markdown("**📄 Detailed Summary:**")
        st.write(summary_data['detailed_summary'])
        
        # Keywords
        st.markdown("**🔑 Keywords:**")
        keywords_html = " ".join([f'<span class="keyword">{kw}</span>' for kw in summary_data['keywords']])
        st.markdown(keywords_html, unsafe_allow_html=True)
        
        # Tags
        st.markdown("**🏷️ Tags:**")
        tags_html = " ".join([f'<span class="tag">{tag}</span>' for tag in summary_data['tags']])
        st.markdown(tags_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**📊 Metadata**")
        st.caption(f"Type: {summary_data['source_type'].upper()}")
        st.caption(f"Words: {summary_data['word_count']:,}")
        st.caption(f"Created: {summary_data['timestamp'][:19]}")
        
        if summary_data.get('url'):
            st.caption(f"URL: {summary_data['url'][:30]}...")
        
        if show_remove:
            if st.button(f"🗑️ Remove", key=f"remove_sum_{summary_data['id']}"):
                summary_manager.remove_summary(summary_data['id'])
                st.success("Summary removed!")
                st.rerun()

# [Keep all other tabs (tab1, tab2, tab4, tab5, tab6) exactly as they are in the original code]