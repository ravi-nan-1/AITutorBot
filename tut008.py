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
import hashlib
import json
import pickle
from pathlib import Path
import shutil
from collections import Counter
import string

# Safe NLTK imports with fallbacks
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except:
            pass
    
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    word_tokenize = None
    sent_tokenize = None
    stopwords = None

# Optional YouTube support
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False

# Optional audio support
try:
    import speech_recognition as sr
    from pydub import AudioSegment
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# ========== PERSISTENCE CONFIGURATION ==========
PERSISTENCE_DIR = os.path.join(tempfile.gettempdir(), "tutor_persistence")
os.makedirs(PERSISTENCE_DIR, exist_ok=True)
CACHE_DIR = os.path.join(tempfile.gettempdir(), "tutor_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cpu")

# ========== PERSISTENCE FUNCTIONS ==========
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
        
        # Auto-save on important events
        if level in ["success", "error"]:
            save_session_state()
            
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

# ========== FIXED CONTENT SUMMARIZER CLASS ==========
class ContentSummarizer:
    """Advanced content summarization with keyword extraction and tagging - FIXED VERSION"""
    
    def __init__(self, logger=None):
        self.logger = logger or TutorLogger()
        
        # Initialize stopwords with fallback
        try:
            if NLTK_AVAILABLE and stopwords:
                self.stop_words = set(stopwords.words('english'))
            else:
                raise Exception("NLTK not available")
        except:
            # Fallback to manual stopwords
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 
                'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'he', 
                'she', 'his', 'her', 'him', 'i', 'you', 'we', 'us', 'my', 'your', 'our'
            }
    
    def safe_tokenize_words(self, text: str) -> List[str]:
        """Safely tokenize words with NLTK fallback"""
        try:
            if NLTK_AVAILABLE and word_tokenize:
                return word_tokenize(text.lower())
            else:
                # Fallback: simple regex-based tokenization
                words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
                return words
        except Exception:
            # Ultimate fallback: split by whitespace and clean
            words = text.lower().split()
            return [re.sub(r'[^a-zA-Z]', '', word) for word in words if re.sub(r'[^a-zA-Z]', '', word)]
    
    def safe_tokenize_sentences(self, text: str) -> List[str]:
        """Safely tokenize sentences with NLTK fallback"""
        try:
            if NLTK_AVAILABLE and sent_tokenize:
                return sent_tokenize(text)
            else:
                # Fallback: split by periods, exclamation marks, and question marks
                sentences = re.split(r'[.!?]+', text)
                return [s.strip() for s in sentences if len(s.strip()) > 10]
        except Exception:
            # Ultimate fallback: split by periods
            sentences = text.split('.')
            return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract main keywords from text with improved error handling"""
        try:
            if not text or len(text.strip()) < 10:
                self.logger.log("⚠️ Text too short for keyword extraction", "warning")
                return []
            
            # Clean and prepare text
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if len(text) < 20:
                return []
            
            # Tokenize words safely
            words = self.safe_tokenize_words(text)
            
            # Filter words
            filtered_words = []
            for word in words:
                if (word and 
                    len(word) > 2 and 
                    len(word) < 20 and 
                    word.isalpha() and 
                    word not in self.stop_words and 
                    not word.isdigit()):
                    filtered_words.append(word)
            
            if not filtered_words:
                self.logger.log("⚠️ No valid keywords found after filtering", "warning")
                return []
            
            # Count word frequency
            word_freq = Counter(filtered_words)
            
            # Get most common keywords
            keywords = [word for word, freq in word_freq.most_common(num_keywords * 2)]
            
            # Prefer longer, more meaningful words
            good_keywords = [kw for kw in keywords if len(kw) > 3]
            if len(good_keywords) < num_keywords:
                good_keywords.extend([kw for kw in keywords if kw not in good_keywords])
            
            result = good_keywords[:num_keywords]
            self.logger.log(f"✅ Extracted {len(result)} keywords", "success")
            return result
            
        except Exception as e:
            self.logger.log(f"❌ Error extracting keywords: {str(e)[:100]}", "error")
            # Emergency fallback: extract simple words
            try:
                simple_words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
                unique_words = list(set(simple_words))[:num_keywords]
                return unique_words
            except:
                return []
    
    def generate_tags(self, text: str, keywords: List[str], num_tags: int = 8) -> List[str]:
        """Generate descriptive tags from content with improved logic"""
        try:
            if not text or len(text.strip()) < 20:
                return keywords[:num_tags] if keywords else []
            
            # Start with keywords as base tags
            base_tags = keywords[:num_tags//2] if keywords else []
            
            # Extract phrases safely
            sentences = self.safe_tokenize_sentences(text[:2000])
            
            phrases = []
            for sentence in sentences[:10]:  # Limit processing
                # Clean sentence
                clean_sentence = re.sub(r'[^\w\s]', ' ', sentence.lower())
                clean_sentence = re.sub(r'\s+', ' ', clean_sentence).strip()
                
                if len(clean_sentence) < 10:
                    continue
                
                words = self.safe_tokenize_words(clean_sentence)
                words = [w for w in words if w and len(w) > 2 and w not in self.stop_words]
                
                # Create 2-word phrases
                for i in range(len(words) - 1):
                    if i < len(words) - 1:
                        phrase = f"{words[i]} {words[i+1]}"
                        if 5 < len(phrase) < 30:
                            phrases.append(phrase)
                
                # Create 3-word phrases
                for i in range(len(words) - 2):
                    if i < len(words) - 2:
                        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                        if 8 < len(phrase) < 40:
                            phrases.append(phrase)
            
            # Get most common phrases
            phrase_tags = []
            if phrases:
                phrase_freq = Counter(phrases)
                phrase_tags = [phrase for phrase, freq in phrase_freq.most_common(num_tags//2) if freq > 1]
            
            # Combine tags
            all_tags = []
            seen = set()
            
            # Add base tags
            for tag in base_tags:
                if tag and tag.lower() not in seen:
                    all_tags.append(tag)
                    seen.add(tag.lower())
            
            # Add phrase tags
            for tag in phrase_tags:
                if tag and tag.lower() not in seen and len(all_tags) < num_tags:
                    all_tags.append(tag)
                    seen.add(tag.lower())
            
            # Fill remaining with keywords if needed
            for kw in keywords:
                if kw and kw.lower() not in seen and len(all_tags) < num_tags:
                    all_tags.append(kw)
                    seen.add(kw.lower())
            
            result = all_tags[:num_tags]
            self.logger.log(f"✅ Generated {len(result)} tags", "success")
            return result
            
        except Exception as e:
            self.logger.log(f"❌ Error generating tags: {str(e)[:100]}", "error")
            return keywords[:num_tags] if keywords else []
    
    def create_two_liner(self, text: str) -> str:
        """Create a compelling 2-line summary with better error handling"""
        try:
            if not text or len(text.strip()) < 20:
                return "Brief content overview.\nKey information and insights provided."
            
            # Clean text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Get sentences safely
            sentences = self.safe_tokenize_sentences(text[:2000])
            
            if not sentences:
                return "Content overview and analysis.\nKey insights and main concepts covered."
            
            # Filter sentences by length
            good_sentences = []
            for sent in sentences:
                sent = sent.strip()
                if 20 <= len(sent) <= 150:  # Good sentence length
                    good_sentences.append(sent)
            
            if len(good_sentences) >= 2:
                line1 = good_sentences[0]
                line2 = good_sentences[1]
            elif len(good_sentences) == 1:
                line1 = good_sentences[0]
                line2 = "Contains detailed information and comprehensive analysis."
            else:
                # Use any sentences we have
                if len(sentences) >= 2:
                    line1 = sentences[0][:120].strip()
                    line2 = sentences[1][:120].strip()
                elif len(sentences) == 1:
                    line1 = sentences[0][:120].strip()
                    line2 = "Provides essential information and key insights."
                else:
                    line1 = text[:120].strip()
                    line2 = "Contains valuable information and analysis."
            
            # Ensure proper endings
            if not line1.endswith(('.', '!', '?')):
                line1 += '.'
            if not line2.endswith(('.', '!', '?')):
                line2 += '.'
            
            result = f"{line1}\n{line2}"
            self.logger.log("✅ Created two-line summary", "success")
            return result
            
        except Exception as e:
            self.logger.log(f"❌ Error creating two-liner: {str(e)[:100]}", "error")
            return "Comprehensive content overview.\nDetailed analysis and key insights provided."
    
    def generate_detailed_summary(self, text: str, source_type: str) -> str:
        """Generate detailed summary using improved rule-based approach"""
        try:
            if not text or len(text.strip()) < 50:
                return "This content provides valuable information on the topic with essential details and key points."
            
            # Clean text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Get sentences safely
            sentences = self.safe_tokenize_sentences(text[:5000])
            
            if len(sentences) <= 3:
                result = " ".join(sentences)
                self.logger.log("✅ Generated summary from few sentences", "success")
                return result
            
            # Get keywords for scoring
            keywords = self.extract_keywords(text, 8)
            keyword_set = set(kw.lower() for kw in keywords)
            
            # Score sentences
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                
                score = 0
                sentence_lower = sentence.lower()
                
                # Position scoring
                if i < 3:  # First sentences are important
                    score += 3
                elif i >= len(sentences) - 2:  # Last sentences
                    score += 2
                
                # Keyword scoring
                keyword_matches = sum(1 for kw in keyword_set if kw in sentence_lower)
                score += keyword_matches * 2
                
                # Length scoring (prefer medium length)
                words_count = len(sentence.split())
                if 8 <= words_count <= 25:
                    score += 2
                elif 25 < words_count <= 35:
                    score += 1
                
                sentence_scores.append((sentence, score, i))
            
            if not sentence_scores:
                return "This content provides comprehensive information on the topic with detailed explanations and key concepts."
            
            # Sort by score and select best sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            selected = sentence_scores[:min(6, len(sentence_scores))]
            
            # Sort selected sentences by original order
            selected.sort(key=lambda x: x[2])
            
            # Create summary
            summary_sentences = [sent[0] for sent in selected]
            summary = " ".join(summary_sentences)
            
            # Ensure minimum length
            if len(summary) < 100 and len(sentence_scores) > 6:
                additional = sentence_scores[6:10]
                additional.sort(key=lambda x: x[2])
                summary += " " + " ".join([sent[0] for sent in additional])
            
            # Limit maximum length
            if len(summary) > 1000:
                summary = summary[:1000].rsplit('.', 1)[0] + '.'
            
            if len(summary) < 50:
                summary = "This content provides comprehensive information on the topic with detailed explanations, key concepts, and valuable insights for learning and understanding."
            
            self.logger.log(f"✅ Generated detailed summary ({len(summary)} chars)", "success")
            return summary
            
        except Exception as e:
            self.logger.log(f"❌ Error in detailed summary: {str(e)[:100]}", "error")
            return "Comprehensive content summary with key insights and main points covered in the material. The content provides detailed information and analysis on the topic."
    
    def generate_summary(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary with metadata and improved error handling"""
        progress_bar = None
        status_text = None
        
        try:
            title = content_data.get('title', 'Unknown Content')
            self.logger.log(f"📊 Starting content analysis for: {title[:50]}...", "info")
            
            text = content_data.get('content', '')
            if not text or len(text.strip()) < 50:
                self.logger.log("❌ Content too short for meaningful analysis", "warning")
                return None
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Extract keywords
                status_text.text("🔍 Extracting main keywords...")
                progress_bar.progress(25)
                keywords = self.extract_keywords(text, 12)
                
                # Step 2: Generate tags
                status_text.text("🏷️ Generating descriptive tags...")
                progress_bar.progress(50)
                tags = self.generate_tags(text, keywords, 10)
                
                # Step 3: Create two-liner
                status_text.text("📝 Creating summary lines...")
                progress_bar.progress(75)
                two_liner = self.create_two_liner(text)
                
                # Step 4: Generate detailed summary
                status_text.text("🧠 Generating detailed summary...")
                progress_bar.progress(90)
                detailed_summary = self.generate_detailed_summary(text, content_data.get('source_type', 'unknown'))
                
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                # Generate unique ID
                content_id = content_data.get('id')
                if not content_id:
                    content_id = hashlib.md5(f"{title}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
                
                summary_data = {
                    'id': content_id,
                    'title': title,
                    'source_type': content_data.get('source_type', 'unknown'),
                    'url': content_data.get('url', ''),
                    'timestamp': datetime.now().isoformat(),
                    'two_liner': two_liner,
                    'detailed_summary': detailed_summary,
                    'keywords': keywords,
                    'tags': tags,
                    'word_count': content_data.get('word_count', len(text.split())),
                    'original_content_preview': text[:500] + "..." if len(text) > 500 else text,
                    'original_title': title
                }
                
                # Clean up progress indicators
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                self.logger.log(f"🎉 Successfully analyzed content: {len(keywords)} keywords, {len(tags)} tags", "success")
                return summary_data
                
            except Exception as processing_error:
                self.logger.log(f"❌ Error during processing: {str(processing_error)[:100]}", "error")
                return None
            
        except Exception as e:
            self.logger.log(f"❌ Error in content analysis: {str(e)[:100]}", "error")
            return None
        finally:
            # Always clean up progress indicators
            try:
                if progress_bar:
                    progress_bar.empty()
                if status_text:
                    status_text.empty()
            except:
                pass

# ========== SUMMARY MANAGER CLASS ==========
class SummaryManager:
    """Manage content summaries and make them available for chat"""
    
    def __init__(self):
        if 'summaries_database' not in st.session_state:
            st.session_state.summaries_database = {
                'summaries': [],
                'statistics': {
                    'total_summaries': 0,
                    'total_keywords': 0,
                    'total_tags': 0
                }
            }
        self.summarizer = ContentSummarizer()
    
    def add_summary(self, content_data: Dict[str, Any]) -> str:
        """Add a summary to the database"""
        try:
            summary_data = self.summarizer.generate_summary(content_data)
            if summary_data:
                # Check if summary for this content already exists
                existing_summaries = st.session_state.summaries_database['summaries']
                existing_titles = [s.get('original_title', s.get('title', '')) for s in existing_summaries]
                
                content_title = content_data.get('title', '')
                if content_title in existing_titles:
                    # Remove existing summary
                    st.session_state.summaries_database['summaries'] = [
                        s for s in existing_summaries 
                        if s.get('original_title', s.get('title', '')) != content_title
                    ]
                
                st.session_state.summaries_database['summaries'].append(summary_data)
                self.update_statistics()
                
                # Also add to content database for chat availability
                self.add_to_chat_context(summary_data)
                
                return summary_data['id']
            return None
        except Exception as e:
            st.error(f"Error adding summary: {e}")
            return None
    
    def add_to_chat_context(self, summary_data: Dict[str, Any]):
        """Make summary available for chat interactions"""
        try:
            # Create a combined text representation for vector storage
            combined_text = f"""
            Title: {summary_data['title']}
            Summary: {summary_data['detailed_summary']}
            Two-line Description: {summary_data['two_liner']}
            Keywords: {', '.join(summary_data['keywords'])}
            Tags: {', '.join(summary_data['tags'])}
            Content Type: {summary_data['source_type']}
            """
            
            # Add to all loaded content for chat availability
            if 'all_loaded_content' not in st.session_state:
                st.session_state.all_loaded_content = []
            
            # Check if this content already exists and remove it
            st.session_state.all_loaded_content = [
                item for item in st.session_state.all_loaded_content 
                if not (item.get('title', '').startswith('SUMMARY:') and 
                       summary_data['title'] in item.get('title', ''))
            ]
            
            # Add new summary content
            st.session_state.all_loaded_content.append({
                'title': f"SUMMARY: {summary_data['title']}",
                'content': combined_text,
                'source_type': 'summary',
                'word_count': len(combined_text.split()),
                'timestamp': datetime.now().isoformat(),
                'url': summary_data.get('url', ''),
                'summary_id': summary_data['id']
            })
                
        except Exception as e:
            st.error(f"Error adding to chat context: {e}")
    
    def get_summary(self, summary_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary by ID"""
        summaries = st.session_state.summaries_database['summaries']
        for summary in summaries:
            if summary.get('id') == summary_id:
                return summary
        return None
    
    def get_all_summaries(self):
        """Get all summaries"""
        return st.session_state.summaries_database['summaries']
    
    def update_statistics(self):
        """Update summary statistics"""
        stats = st.session_state.summaries_database['statistics']
        summaries = st.session_state.summaries_database['summaries']
        
        stats['total_summaries'] = len(summaries)
        stats['total_keywords'] = sum(len(summary.get('keywords', [])) for summary in summaries)
        stats['total_tags'] = sum(len(summary.get('tags', [])) for summary in summaries)
    
    def get_statistics(self):
        """Get summary statistics"""
        return st.session_state.summaries_database['statistics']
    
    def remove_summary(self, summary_id: str):
        """Remove a summary by ID"""
        summaries = st.session_state.summaries_database['summaries']
        st.session_state.summaries_database['summaries'] = [
            summary for summary in summaries if summary.get('id') != summary_id
        ]
        self.update_statistics()
        
        # Also remove from chat context
        if 'all_loaded_content' in st.session_state:
            st.session_state.all_loaded_content = [
                item for item in st.session_state.all_loaded_content 
                if item.get('summary_id') != summary_id
            ]
    
    def search_summaries(self, query: str) -> List[Dict[str, Any]]:
        """Search summaries by keyword or tag"""
        query = query.lower()
        results = []
        
        for summary in st.session_state.summaries_database['summaries']:
            # Search in title
            if query in summary.get('title', '').lower():
                results.append(summary)
                continue
            
            # Search in keywords
            if any(query in keyword.lower() for keyword in summary.get('keywords', [])):
                results.append(summary)
                continue
            
            # Search in tags
            if any(query in tag.lower() for tag in summary.get('tags', [])):
                results.append(summary)
                continue
            
            # Search in content
            if query in summary.get('detailed_summary', '').lower():
                results.append(summary)
                continue
        
        return results

# ========== CONTENT MANAGER WITH PERSISTENCE ==========
class ContentManager:
    """Manage and display stored content with persistence"""
    
    def __init__(self):
        self.persistence_file = os.path.join(PERSISTENCE_DIR, "content_database.json")
        
        if 'content_database' not in st.session_state:
            # Try to load from file first
            loaded_data = self.load_from_disk()
            if loaded_data:
                st.session_state.content_database = loaded_data
            else:
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
    
    def load_from_disk(self):
        """Load content database from disk"""
        try:
            if os.path.exists(self.persistence_file):
                with open(self.persistence_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Error loading content database: {e}")
        return None
    
    def save_to_disk(self):
        """Save content database to disk"""
        try:
            with open(self.persistence_file, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.content_database, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving content database: {e}")
            return False
    
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
        self.save_to_disk()
        save_session_state()
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
        self.save_to_disk()
        save_session_state()
    
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
        self.save_to_disk()
        save_session_state()

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
        
        try:
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
            
        finally:
            # Clean up progress indicators
            try:
                progress_bar.empty()
                status_text.empty()
            except:
                pass
        
        return self.scraped_data

# ========== CONTENT PROCESSORS WITH BETTER ERROR HANDLING ==========
class ContentProcessor:
    def __init__(self, logger=None):
        self.logger = logger or TutorLogger()
    
    def process_pdf(self, pdf_file):
        """Extract text from PDF file with logging"""
        progress_bar = None
        status_text = None
        
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
            
            # Clean up
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
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
        finally:
            try:
                if progress_bar:
                    progress_bar.empty()
                if status_text:
                    status_text.empty()
            except:
                pass

    def process_youtube(self, url):
        """Extract transcript from YouTube video with enhanced error handling"""
        progress_bar = None
        status_text = None
        
        try:
            if not YOUTUBE_AVAILABLE:
                st.error("🚫 YouTube processing not available. Please install youtube-transcript-api")
                st.code("pip install youtube-transcript-api")
                return None
            
            self.logger.log(f"🎥 Processing YouTube video: {url[:50]}...", "info")
            
            # Create progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Enhanced video ID extraction
            def extract_video_id(url):
                import re
                url = url.strip()
                patterns = [
                    r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/watch\?.*&v=)([a-zA-Z0-9_-]{11})',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, url)
                    if match:
                        return match.group(1)
                return None
            
            # Extract video ID
            video_id = extract_video_id(url)
            
            if not video_id:
                self.logger.log(f"❌ Could not extract video ID from URL: {url}", "error")
                st.error("🚫 Invalid YouTube URL format!")
                return None
            
            self.logger.log(f"🎬 Video ID identified: {video_id}", "info")
            
            progress_bar.progress(0.3)
            status_text.text("🔍 Looking for video transcripts...")
            
            try:
                # Get transcript using YouTubeTranscriptApi
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                progress_bar.progress(0.6)
                status_text.text("📝 Processing transcript...")
                
                transcript = None
                available_languages = []
                
                for transcript_obj in transcript_list:
                    available_languages.append(transcript_obj.language)
                    if transcript_obj.language_code.startswith('en'):
                        transcript = transcript_obj.fetch()
                        self.logger.log(f"✅ Found English transcript", "success")
                        break
                
                if not transcript:
                    self.logger.log(f"⚠️ No English transcript found. Available: {', '.join(available_languages[:5])}", "warning")
                    first_transcript = next(iter(transcript_list))
                    transcript = first_transcript.fetch()
                    self.logger.log(f"✅ Using {first_transcript.language} transcript", "success")
                
                progress_bar.progress(0.9)
                status_text.text("📄 Finalizing transcript...")
                
                # Combine transcript text
                text = " ".join([entry['text'] for entry in transcript])
                word_count = len(text.split())
                
                if word_count < 20:
                    self.logger.log("❌ Transcript too short", "error")
                    st.error("Transcript is too short. Please try a longer video.")
                    return None
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Success! Extracted {word_count} words")
                
                # Clean up
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                self.logger.log(f"✅ Successfully processed YouTube video: {word_count} words", "success")
                
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
                error_msg = str(e)
                if "429" in error_msg:
                    self.logger.log("❌ YouTube rate limiting detected", "error")
                    st.error("🚫 YouTube is rate limiting. Please try again later.")
                else:
                    self.logger.log(f"❌ Transcript error: {error_msg[:100]}", "error")
                    st.error("🚫 Could not get transcript. Video may not have captions.")
                return None
                
        except Exception as e:
            self.logger.log(f"❌ Error processing YouTube video: {str(e)}", "error")
            st.error(f"Failed to process YouTube video: {str(e)}")
            return None
        finally:
            try:
                if progress_bar:
                    progress_bar.empty()
                if status_text:
                    status_text.empty()
            except:
                pass
    
    def process_audio(self, audio_file):
        """Extract text from audio file using speech recognition with logging"""
        progress_bar = None
        status_text = None
        temp_path = None
        wav_path = None
        
        try:
            if not AUDIO_AVAILABLE:
                st.error("🚫 Audio processing not available. Please install required packages")
                st.code("pip install speechrecognition pydub")
                return None
            
            self.logger.log(f"🎵 Processing audio file: {audio_file.name}", "info")
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            file_size_mb = len(audio_file.getvalue()) / (1024 * 1024)
            self.logger.log(f"📊 File size: {file_size_mb:.2f} MB", "info")
            
            # Check file size limit
            if file_size_mb > 25:  # 25MB limit
                self.logger.log(f"❌ File too large: {file_size_mb:.2f} MB", "error")
                st.error("File too large. Please use files smaller than 25MB.")
                return None
            
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
            
            # Check duration limit
            if duration_seconds > 600:  # 10 minutes
                self.logger.log(f"❌ Audio too long: {duration_seconds:.1f} seconds", "error")
                st.error("Audio too long. Please use files shorter than 10 minutes.")
                return None
            
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
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
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
        finally:
            # Clean up temporary files
            try:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)
                if progress_bar:
                    progress_bar.empty()
                if status_text:
                    status_text.empty()
                self.logger.log("🧹 Cleaned up temporary files", "info")
            except:
                pass
    
    def process_text(self, text_input):
        """Process plain text input"""
        try:
            self.logger.log("📝 Processing text input...", "info")
            
            if not text_input or len(text_input.strip()) < 10:
                self.logger.log("❌ Text input too short", "error")
                return None
            
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
    def __init__(self, vectorstore, llm, content_manager=None, summary_manager=None):
        self.vectorstore = vectorstore
        self.llm = llm
        self.conversation_history = []
        self.all_content = []
        self.content_manager = content_manager or ContentManager()
        self.summary_manager = summary_manager or SummaryManager()
    
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
            
            # Also check summaries
            summary_results = self.summary_manager.search_summaries(question)
            has_relevant_summaries = len(summary_results) > 0
            
            # Search for relevant documents
            docs = self.vectorstore.similarity_search(question, k=6)
            
            # If no relevant docs found and content doesn't exist
            if (not docs or len(docs) == 0) and not content_exists and not has_relevant_summaries:
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
            
            # If no relevant documents found but we have summaries
            if not relevant_docs and has_relevant_summaries:
                # Use summary information for response
                summary_context = "\n".join([
                    f"Summary: {summary['detailed_summary']}\nKeywords: {', '.join(summary['keywords'])}"
                    for summary in summary_results[:3]
                ])
                
                prompt = f"Based on these summaries: {summary_context}\nQuestion: {question}\nAnswer:"
                response = self.llm._call(prompt)
                
                return {
                    "answer": response,
                    "sources": [{"title": f"Summary: {s['title']}", "type": "summary", "snippet": s['two_liner']} for s in summary_results[:3]],
                    "suggestions": self.generate_summary_suggestions(summary_results),
                    "context_used": f"Summaries: {len(summary_results)} relevant summaries found"
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
    
    def generate_summary_suggestions(self, summaries: List[Dict]) -> List[str]:
        suggestions = []
        
        if summaries:
            # Use keywords from summaries for suggestions
            all_keywords = []
            for summary in summaries[:2]:
                all_keywords.extend(summary.get('keywords', [])[:3])
            
            unique_keywords = list(set(all_keywords))[:3]
            if unique_keywords:
                suggestions.append(f"Explain {unique_keywords[0]} in detail")
                if len(unique_keywords) > 1:
                    suggestions.append(f"What about {unique_keywords[1]}?")
            
            suggestions.append("Show me more summaries")
            suggestions.append("Give me a quiz on this topic")
        
        return suggestions[:4]

# ========== HELPER FUNCTION WITH PERSISTENCE ==========
def process_loaded_content(content_data, logger, content_manager, summary_manager=None):
    """Process and vectorize loaded content with logging and persistence"""
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
        
        # Vector store persistence
        vectorstore_path = os.path.join(PERSISTENCE_DIR, "vectorstore")
        
        # Create or update vector store
        logger.log("🔗 Creating vector embeddings...", "info")
        if os.path.exists(vectorstore_path) and "vectorstore" not in st.session_state:
            # Load existing vector store
            logger.log("📂 Loading existing vector store...", "info")
            st.session_state.vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            # Add new documents
            if texts:
                new_vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
                st.session_state.vectorstore.merge_from(new_vectorstore)
            logger.log("✅ Updated existing vector store", "success")
        elif "vectorstore" in st.session_state:
            # Update existing in-memory vector store
            new_vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
            st.session_state.vectorstore.merge_from(new_vectorstore)
            logger.log("✅ Updated existing vector store", "success")
        else:
            # Create new vector store
            st.session_state.vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
            logger.log("✅ Created new vector store", "success")
        
        # Save vector store to disk
        st.session_state.vectorstore.save_local(vectorstore_path)
        logger.log("💾 Vector store saved to disk", "success")
        
        # Initialize LLM and Tutor
        if "llm" not in st.session_state:
            logger.log("🤖 Initializing language model...", "info")
            st.session_state.llm = TutorLLM()
            logger.log("✅ Language model ready", "success")
        
        logger.log("🎓 Initializing tutor system...", "info")
        st.session_state.tutor = RAGTutor(st.session_state.vectorstore, st.session_state.llm, content_manager, summary_manager)
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
        
        # Save session state
        save_session_state()
        
        logger.log(f"🎉 Successfully processed all content! Ready for Q&A", "success")
        st.success(f"✅ Successfully loaded {len(texts)} text chunks!")
        st.info("Go to 'Chat & Learn' tab to start learning!")
        
    except Exception as e:
        logger.log(f"❌ Error processing content: {str(e)[:200]}", "error")
        st.error(f"Error processing content: {e}")

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="AI Tutor Assistant", layout="wide", page_icon="🎓")

# Load previous session state on startup
if not load_session_state():
    st.sidebar.info("🔍 Starting fresh session...")
else:
    st.sidebar.success("♻️ Previous session restored!")

# Custom CSS (keeping it short for space)
st.markdown("""
<style>
    .summary-card {
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f1f8e9;
    }
    .tag {
        display: inline-block;
        background-color: #e3f2fd;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem;
        border-radius: 15px;
        font-size: 0.8em;
        border: 1px solid #2196F3;
    }
    .keyword {
        display: inline-block;
        background-color: #fff3e0;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem;
        border-radius: 10px;
        font-size: 0.8em;
        border: 1px solid #FF9800;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
logger = TutorLogger()
content_manager = ContentManager()
summary_manager = SummaryManager()

st.title("🎓 AI Tutor - Your Learning Assistant")

# Initialize session state
if "all_loaded_content" not in st.session_state:
    st.session_state.all_loaded_content = []

# Sidebar controls (simplified for space)
with st.sidebar:
    st.header("⚙️ Settings")
    
    learning_mode = st.selectbox("Learning Mode:", ["Explain", "Simplify", "Elaborate", "Summarize", "Quiz"])
    difficulty = st.selectbox("Difficulty:", ["Beginner", "Medium", "Advanced"])
    max_pages = st.slider("Pages to analyze:", 5, 20, 10)
    
    # Show statistics if available
    stats = content_manager.get_statistics()
    summary_stats = summary_manager.get_statistics()
    
    if stats['total_documents'] > 0 or summary_stats['total_summaries'] > 0:
        st.markdown("### 📊 Content Statistics")
        if stats['total_documents'] > 0:
            st.metric("Documents", stats['total_documents'])
            st.metric("Words", f"{stats['total_words']:,}")
        if summary_stats['total_summaries'] > 0:
            st.metric("Summaries", summary_stats['total_summaries'])

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat & Learn", "📚 Load Content", "📊 Summarize Content", "🗂️ Content Library", "📋 Activity Logs"])

# Tab 3: Content Summarization (Main fix)
with tab3:
    st.header("📊 Content Summarization")
    st.markdown("Generate comprehensive summaries with keywords and tags from your content")
    
    # Get all available content
    all_documents = content_manager.get_all_documents()
    all_summaries = summary_manager.get_all_summaries()
    
    if not all_documents:
        st.info("📚 No content available for summarization. Please load content first in the 'Load Content' tab.")
    else:
        # Content selection
        content_options = {f"{doc['title']} ({doc['source_type']})": doc for doc in all_documents}
        selected_content_key = st.selectbox(
            "Choose content to summarize:",
            options=list(content_options.keys()),
            help="Select from your loaded content"
        )
        
        selected_content = content_options[selected_content_key]
        
        # Generate summary button
        if st.button("🚀 Generate Comprehensive Summary", type="primary", use_container_width=True):
            with st.spinner("Analyzing content and generating summary..."):
                summary_id = summary_manager.add_summary(selected_content)
                
                if summary_id:
                    st.success("✅ Summary generated successfully!")
                    
                    # Show the generated summary
                    summary_data = summary_manager.get_summary(summary_id)
                    if summary_data:
                        st.subheader("📋 Summary Results")
                        
                        with st.container():
                            st.markdown(f'<div class="summary-card">', unsafe_allow_html=True)
                            
                            st.markdown(f"### {summary_data['title']}")
                            st.markdown(f"**Type:** {summary_data['source_type'].upper()} | **Words:** {summary_data['word_count']:,}")
                            
                            st.markdown("---")
                            
                            # Two-liner
                            st.markdown("#### 📝 Two-Line Summary")
                            st.info(summary_data['two_liner'])
                            
                            # Detailed summary
                            st.markdown("#### 📄 Detailed Summary")
                            st.write(summary_data['detailed_summary'])
                            
                            # Keywords
                            st.markdown("#### 🔑 Main Keywords")
                            keywords_html = " ".join([f'<span class="keyword">{kw}</span>' for kw in summary_data['keywords']])
                            st.markdown(keywords_html, unsafe_allow_html=True)
                            
                            # Tags
                            st.markdown("#### 🏷️ Content Tags")
                            tags_html = " ".join([f'<span class="tag">{tag}</span>' for tag in summary_data['tags']])
                            st.markdown(tags_html, unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.info("💡 This summary is now available in the chat! You can ask questions about this content.")
                        
                else:
                    st.error("❌ Failed to generate summary. Please try again with different content.")
    
    # Display existing summaries
    if all_summaries:
        st.markdown("---")
        st.subheader("📂 Existing Summaries")
        
        for summary in reversed(all_summaries):  # Show newest first
            with st.expander(f"📊 {summary['title']} ({summary['source_type']})", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("**Two-Line Summary:**")
                    st.info(summary['two_liner'])
                    
                    st.markdown("**Keywords:**")
                    keywords_html = " ".join([f'<span class="keyword">{kw}</span>' for kw in summary['keywords'][:8]])
                    st.markdown(keywords_html, unsafe_allow_html=True)
                
                with col2:
                    st.caption(f"Words: {summary['word_count']:,}")
                    if st.button(f"🗑️ Remove", key=f"remove_summary_{summary['id']}"):
                        summary_manager.remove_summary(summary['id'])
                        st.success("Summary removed!")
                        st.rerun()

# Tab 2: Load Content
with tab2:
    st.header("📚 Load Learning Content")
    
    content_type = st.radio("Select content type:", ["🌐 Website", "📄 PDF", "🎥 YouTube", "🎵 Audio", "📝 Text"], horizontal=True)
    
    content_processor = ContentProcessor(logger=logger)
    
    if content_type == "🌐 Website":
        url_input = st.text_input("Enter URL:", placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence")
        if st.button("🔍 Load Website", type="primary") and url_input:
            logger.clear_logs()
            with st.spinner("Loading website content..."):
                crawler = TutorWebCrawler(max_pages=max_pages, logger=logger)
                scraped_data = crawler.crawl_website(url_input)
                
                if scraped_data:
                    for doc in scraped_data:
                        content_manager.add_document(doc)
                    st.session_state.all_loaded_content.extend(scraped_data)
                    process_loaded_content(scraped_data, logger, content_manager, summary_manager)
    
    elif content_type == "📄 PDF":
        pdf_file = st.file_uploader("Upload PDF file", type=['pdf'])
        if pdf_file and st.button("📄 Load PDF", type="primary"):
            logger.clear_logs()
            with st.spinner("Processing PDF..."):
                pdf_data = content_processor.process_pdf(pdf_file)
                if pdf_data:
                    content_manager.add_document(pdf_data)
                    st.session_state.all_loaded_content.append(pdf_data)
                    process_loaded_content([pdf_data], logger, content_manager, summary_manager)
    
    elif content_type == "🎥 YouTube":
        youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
        if youtube_url and st.button("🎥 Load Video", type="primary"):
            logger.clear_logs()
            video_data = content_processor.process_youtube(youtube_url)
            if video_data:
                content_manager.add_document(video_data)
                st.session_state.all_loaded_content.append(video_data)
                process_loaded_content([video_data], logger, content_manager, summary_manager)
    
    elif content_type == "🎵 Audio":
        audio_file = st.file_uploader("Upload audio file", type=['mp3', 'wav', 'ogg', 'm4a'])
        if audio_file and st.button("🎵 Process Audio", type="primary"):
            logger.clear_logs()
            audio_data = content_processor.process_audio(audio_file)
            if audio_data:
                content_manager.add_document(audio_data)
                st.session_state.all_loaded_content.append(audio_data)
                process_loaded_content([audio_data], logger, content_manager, summary_manager)
    
    elif content_type == "📝 Text":
        text_input = st.text_area("Enter text content:", height=200, placeholder="Paste your learning material here...")
        if text_input and st.button("📝 Load Text", type="primary"):
            logger.clear_logs()
            text_data = content_processor.process_text(text_input)
            if text_data:
                content_manager.add_document(text_data)
                st.session_state.all_loaded_content.append(text_data)
                process_loaded_content([text_data], logger, content_manager, summary_manager)

# Tab 1: Chat Interface (simplified)
with tab1:
    if "content_loaded" not in st.session_state:
        st.info("👈 Please load content from the 'Load Content' tab first")
        if st.session_state.all_loaded_content:
            st.success(f"📚 Found {len(st.session_state.all_loaded_content)} previously loaded documents!")
            if st.button("🔄 Restore Previous Session"):
                process_loaded_content(st.session_state.all_loaded_content, logger, content_manager, summary_manager)
                st.rerun()
        st.stop()
    
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your AI tutor. What would you like to learn about?"
        })
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the loaded content..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                tutor = st.session_state.tutor
                response = tutor.get_response(prompt, mode=learning_mode.lower(), difficulty=difficulty.lower())
                st.write(response["answer"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", []),
                    "suggestions": response.get("suggestions", [])
                })
        
        save_session_state()
        st.rerun()

# Remaining tabs (simplified for space)
with tab4:
    st.header("🗂️ Content Library")
    documents = content_manager.get_all_documents()
    summaries = summary_manager.get_all_summaries()
    
    if documents or summaries:
        st.success(f"📚 Total: {len(documents)} documents, {len(summaries)} summaries")
    else:
        st.info("No content loaded yet.")

with tab5:
    st.header("📋 Activity Logs")
    logs = logger.get_logs(limit=20)
    for log in reversed(logs):
        st.caption(f"{log['timestamp']} - {log['message']}")