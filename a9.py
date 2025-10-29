import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from typing import Optional, List, Dict, Any
import urllib3
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from pydantic import Field
import re
import time
from urllib.parse import urljoin, urlparse, parse_qs
from collections import deque
from datetime import datetime
import concurrent.futures
from threading import Lock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [Keep the RobustWebCrawler class exactly the same as before]

class RobustWebCrawler:
    """Robust web crawler with better error handling"""
    
    def __init__(self, max_depth=2, max_pages=50, timeout=15):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.timeout = timeout
        self.visited_urls = set()
        self.visited_lock = Lock()
        self.scraped_data = []
        self.data_lock = Lock()
        self.failed_urls = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
    def normalize_url(self, url):
        """Normalize URL to avoid duplicates"""
        # Remove fragment and trailing slash
        url = url.split('#')[0].rstrip('/')
        
        # Handle Wikipedia special cases
        if 'wikipedia.org' in url:
            # Remove mobile subdomain
            url = url.replace('//m.', '//')
            url = url.replace('//mobile.', '//')
            # Remove edit parameters
            url = re.sub(r'[?&](action|oldid|diff)=[^&]*', '', url)
        
        # Remove common tracking parameters
        parsed = urlparse(url)
        if parsed.query:
            query_params = parse_qs(parsed.query)
            filtered_params = {k: v for k, v in query_params.items() 
                             if k not in ['utm_source', 'utm_medium', 'utm_campaign', 'ref']}
            if filtered_params:
                query = '&'.join([f"{k}={v[0]}" for k, v in filtered_params.items()])
                url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{query}"
            else:
                url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        return url
    
    def is_valid_url(self, url, base_domain):
        """Check if URL should be crawled"""
        try:
            parsed = urlparse(url)
            base_parsed = urlparse(base_domain)
            
            # Must be same domain
            if parsed.netloc != base_parsed.netloc:
                return False
            
            # Skip non-content URLs
            skip_patterns = [
                r'\.(jpg|jpeg|png|gif|svg|pdf|zip|exe|mp3|mp4|css|js)(\?|$)',
                r'/Special:',  # Wikipedia special pages
                r'/Talk:',     # Wikipedia talk pages
                r'/User:',     # Wikipedia user pages
                r'/File:',     # Wikipedia file pages
                r'#cite',      # Citations
                r'/index\.php\?',  # PHP index pages
                r'(login|signin|register|logout)',
                r'javascript:',
                r'mailto:',
            ]
            
            url_lower = url.lower()
            for pattern in skip_patterns:
                if re.search(pattern, url, re.I):
                    return False
            
            return True
        except Exception as e:
            logging.error(f"Error validating URL {url}: {e}")
            return False
    
    def extract_content(self, url):
        """Extract content from URL with robust error handling"""
        try:
            # Make request
            response = self.session.get(url, timeout=self.timeout, verify=False, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return None
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser', from_encoding=response.encoding)
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "No Title"
            
            # Remove non-content elements
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 
                                        'aside', 'form', 'noscript', 'svg']):
                element.decompose()
            
            # Wikipedia-specific handling
            if 'wikipedia.org' in url:
                # Remove Wikipedia UI elements
                for selector in ['.navbox', '.sidebar', '.ambox', '.ombox', '.mbox', 
                               '.hatnote', '.reference', '.reflist']:
                    for element in soup.select(selector):
                        element.decompose()
                
                # Get main content - but keep infobox for biographical data
                content_div = soup.find('div', {'id': 'mw-content-text'})
                if content_div:
                    # Extract infobox data separately if exists
                    infobox_text = ""
                    infobox = soup.find('table', {'class': 'infobox'})
                    if infobox:
                        infobox_text = infobox.get_text(separator=' | ', strip=True)
                    
                    content = content_div.get_text(separator='\n', strip=True)
                    if infobox_text:
                        content = f"INFOBOX DATA: {infobox_text}\n\nMAIN CONTENT:\n{content}"
                else:
                    content = soup.get_text(separator='\n', strip=True)
            else:
                # General content extraction
                # Try to find main content areas
                main_content = None
                for selector in ['main', 'article', '[role="main"]', '.main-content', 
                               '#content', '.content', '.post-content']:
                    element = soup.select_one(selector)
                    if element:
                        main_content = element.get_text(separator='\n', strip=True)
                        break
                
                content = main_content if main_content else soup.get_text(separator='\n', strip=True)
            
            # Clean content
            lines = []
            for line in content.splitlines():
                line = line.strip()
                if line and len(line) > 20 and not line.startswith(('©', 'Cookie', 'Privacy')):
                    lines.append(line)
            
            cleaned_content = '\n'.join(lines[:1000])  # Limit lines
            
            # Extract links
            links = set()
            for a_tag in soup.find_all('a', href=True):
                href = a_tag.get('href', '')
                if href:
                    absolute_url = urljoin(url, href)
                    normalized = self.normalize_url(absolute_url)
                    if self.is_valid_url(normalized, url):
                        links.add(normalized)
            
            return {
                'url': url,
                'title': title[:200],
                'content': cleaned_content[:5000],  # Limit content size
                'links': list(links)[:100],  # Limit links
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.Timeout:
            logging.error(f"Timeout for {url}")
            self.failed_urls.add(url)
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error for {url}: {e}")
            self.failed_urls.add(url)
            return None
        except Exception as e:
            logging.error(f"Unexpected error for {url}: {e}")
            self.failed_urls.add(url)
            return None
    
    def crawl_page(self, url, depth):
        """Crawl a single page"""
        # Check if should crawl
        with self.visited_lock:
            if url in self.visited_urls or url in self.failed_urls:
                return []
            if len(self.scraped_data) >= self.max_pages:
                return []
            self.visited_urls.add(url)
        
        # Extract content
        data = self.extract_content(url)
        
        if data and data['content']:
            with self.data_lock:
                self.scraped_data.append(data)
            
            # Return links for further crawling
            if depth < self.max_depth:
                return [(link, depth + 1) for link in data['links']]
        
        return []
    
    def crawl_website(self, start_url, progress_callback=None):
        """Main crawling function with thread pool"""
        # Initialize
        start_url = self.normalize_url(start_url)
        url_queue = deque([(start_url, 0)])
        processed = 0
        
        # Use thread pool for concurrent crawling
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            while url_queue and len(self.scraped_data) < self.max_pages:
                # Get batch of URLs to process
                batch = []
                for _ in range(min(5, len(url_queue))):
                    if url_queue:
                        batch.append(url_queue.popleft())
                
                if not batch:
                    break
                
                # Submit crawl tasks
                futures = []
                for url, depth in batch:
                    future = executor.submit(self.crawl_page, url, depth)
                    futures.append(future)
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        new_urls = future.result()
                        url_queue.extend(new_urls)
                    except Exception as e:
                        logging.error(f"Error in future: {e}")
                
                # Update progress
                processed += len(batch)
                if progress_callback:
                    with self.data_lock:
                        scraped_count = len(self.scraped_data)
                    progress_callback(
                        scraped_count,
                        len(self.visited_urls),
                        batch[-1][0] if batch else start_url
                    )
                
                # Small delay
                time.sleep(0.2)
        
        return self.scraped_data

# Enhanced embeddings for better retrieval
class EnhancedEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    @torch.no_grad()
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        # Process in batches
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_texts = [text[:512] for text in batch_texts]  # Truncate
            
            encoded = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors='pt'
            ).to(DEVICE)
            
            model_output = self.model(**encoded)
            pooled = self.mean_pooling(model_output, encoded['attention_mask'])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.extend(normalized.cpu().numpy().tolist())
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Enhanced LLM with better prompting
class EnhancedLLM(LLM):
    model_name: str = Field(default="google/flan-t5-large")  # Using larger model
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(DEVICE)
        self.model.eval()
    
    @property
    def _llm_type(self) -> str:
        return "enhanced_llm"
    
    @torch.no_grad()
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        ).to(DEVICE)
        
        # Generate with better parameters
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            min_length=100,  # Ensure detailed answers
            num_beams=5,
            temperature=0.8,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.2,
            length_penalty=1.5,  # Encourage longer answers
            early_stopping=False
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def create_enhanced_prompt(question: str, context: str, doc_metadata: List[dict]) -> str:
    """Create a comprehensive prompt for better answers"""
    
    # Analyze question type
    question_lower = question.lower()
    is_who_question = any(word in question_lower for word in ['who is', 'who was', 'who are', 'who were'])
    is_what_question = any(word in question_lower for word in ['what is', 'what was', 'what are', 'what were'])
    is_biographical = any(word in question_lower for word in ['biography', 'life', 'born', 'died', 'person'])
    
    # Create appropriate prompt based on question type
    if is_who_question or is_biographical:
        prompt = f"""You are a knowledgeable assistant. Answer the biographical question comprehensively.

Context from sources:
{context}

Question: {question}

Provide a detailed answer that includes:
1. Full name and basic identity
2. Birth and death dates (if applicable)
3. Nationality and occupation
4. Major achievements and contributions
5. Historical significance and legacy
6. Any other relevant biographical information

Write a complete, well-structured response:"""

    elif is_what_question:
        prompt = f"""You are a knowledgeable assistant. Provide a comprehensive explanation.

Context from sources:
{context}

Question: {question}

Provide a detailed answer that includes:
1. Clear definition or explanation
2. Key characteristics or features
3. Historical context or background
4. Importance or significance
5. Examples if relevant

Write a complete, informative response:"""

    else:
        # General comprehensive prompt
        prompt = f"""You are a knowledgeable assistant with access to detailed information.

Context from sources:
{context}

Question: {question}

Instructions:
- Provide a comprehensive, detailed answer based on the context
- Include all relevant information from the sources
- Structure your response clearly with main points
- Be specific and include examples where appropriate
- If the context contains biographical data, include all key details
- Aim for completeness rather than brevity

Detailed answer:"""
    
    return prompt

# Streamlit UI
st.set_page_config(page_title="Enhanced Web Crawler & Q&A", layout="wide")
st.title("🔍 Enhanced Web Crawler & Q&A Bot")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    max_pages = st.slider("Max pages to crawl", 10, 100, 50)
    max_depth = st.slider("Max crawl depth", 1, 3, 2)
    
    st.subheader("Model Selection")
    model_size = st.selectbox(
        "LLM Model Size",
        ["Base (Faster)", "Large (Better Quality)"],
        index=1
    )
    
    if model_size == "Base (Faster)":
        model_name = "google/flan-t5-base"
    else:
        model_name = "google/flan-t5-large"
    
    st.subheader("Answer Settings")
    num_sources = st.slider("Number of sources to use", 3, 10, 6)
    answer_style = st.selectbox(
        "Answer Style",
        ["Comprehensive", "Concise", "Academic"],
        index=0
    )

# Main tabs
tab1, tab2 = st.tabs(["🕸️ Crawler", "💬 Enhanced Q&A"])

with tab1:
    url_input = st.text_input(
        "Enter URL to crawl:", 
        placeholder="https://en.wikipedia.org/wiki/Bhagat_Singh"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        crawl_button = st.button("Start Crawling", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear Data", use_container_width=True):
            if os.path.exists("enhanced_index"):
                import shutil
                shutil.rmtree("enhanced_index")
            for key in ["vectorstore", "crawl_info", "messages"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    if crawl_button and url_input:
        start_time = time.time()
        
        # Initialize crawler
        crawler = RobustWebCrawler(max_depth=max_depth, max_pages=max_pages)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(pages_crawled, total_visited, current_url):
            progress = pages_crawled / max_pages if max_pages > 0 else 0
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"✅ Crawled: {pages_crawled}/{max_pages} | Current: {current_url[:60]}...")
        
        # Crawl website
        with st.spinner("🔍 Crawling website..."):
            scraped_data = crawler.crawl_website(url_input, progress_callback=update_progress)
        
        elapsed_time = time.time() - start_time
        
        if scraped_data:
            st.success(f"✅ Successfully crawled {len(scraped_data)} pages in {elapsed_time:.1f} seconds!")
            
            # Index content with better chunking
            with st.spinner("📚 Creating enhanced index..."):
                # Initialize embeddings
                embeddings = EnhancedEmbeddings()
                
                # Prepare texts with better chunking strategy
                texts = []
                metadata = []
                
                # Use smaller chunks with more overlap for better retrieval
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
                )
                
                for page in scraped_data:
                    # Create rich content with title context
                    page_content = f"Page Title: {page['title']}\nURL: {page['url']}\n\n{page['content']}"
                    chunks = splitter.split_text(page_content)
                    
                    for i, chunk in enumerate(chunks[:15]):  # More chunks per page
                        if len(chunk) > 50:
                            texts.append(chunk)
                            metadata.append({
                                'url': page['url'],
                                'title': page['title'],
                                'chunk_index': i
                            })
                
                # Create and save index
                vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
                vectorstore.save_local("enhanced_index")
                
                st.session_state.vectorstore = vectorstore
                st.session_state.crawl_info = {
                    'pages': len(scraped_data),
                    'chunks': len(texts),
                    'domain': urlparse(url_input).netloc
                }
                
                st.success(f"✅ Created enhanced index with {len(texts)} text chunks!")

with tab2:
    st.header("Enhanced Q&A Bot")
    
    # Load index if exists
    if "vectorstore" not in st.session_state:
        if os.path.exists("enhanced_index"):
            with st.spinner("Loading enhanced index..."):
                embeddings = EnhancedEmbeddings()
                st.session_state.vectorstore = FAISS.load_local(
                    "enhanced_index",
                    embeddings,
                    allow_dangerous_deserialization=True
                )
        else:
            st.warning("No indexed content found. Please crawl a website first!")
            st.stop()
    
    # Initialize LLM
    if "llm" not in st.session_state:
        with st.spinner(f"Loading {model_name} model..."):
            st.session_state.llm = EnhancedLLM(model_name=model_name)
    
    # Show info
    if "crawl_info" in st.session_state:
        info = st.session_state.crawl_info
        st.info(f"📊 Indexed: {info['pages']} pages, {info['chunks']} chunks from {info['domain']}")
    
    # Example questions
    st.markdown("**Example questions:**")
    example_cols = st.columns(3)
    examples = [
        "Who is Bhagat Singh?",
        "What were his major contributions?",
        "Tell me about his life and legacy"
    ]
    for i, col in enumerate(example_cols):
        if i < len(examples):
            with col:
                if st.button(examples[i], key=f"ex_{i}"):
                    st.session_state.next_question = examples[i]
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Handle example question
    if "next_question" in st.session_state:
        prompt = st.session_state.next_question
        del st.session_state.next_question
    else:
        prompt = st.chat_input("Ask a comprehensive question...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Researching and formulating comprehensive answer..."):
                # Enhanced retrieval with more sources
                docs = st.session_state.vectorstore.similarity_search(
                    prompt, 
                    k=num_sources
                )
                
                if docs:
                    # Combine contexts intelligently
                    contexts = []
                    seen_content = set()
                    sources = {}
                    
                    for doc in docs:
                        content = doc.page_content.strip()
                        # Avoid duplicate content
                        content_hash = hash(content[:100])
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            contexts.append(content)
                            
                            url = doc.metadata.get('url', '')
                            if url:
                                sources[url] = doc.metadata.get('title', '')
                    
                    # Create rich context
                    combined_context = "\n\n---\n\n".join(contexts)
                    
                    # Generate comprehensive answer
                    enhanced_prompt = create_enhanced_prompt(
                        prompt, 
                        combined_context[:3000],  # Use more context
                        list(docs)
                    )
                    
                    response = st.session_state.llm._call(enhanced_prompt)
                    
                    # Format response with sources
                    formatted_response = f"{response}\n\n"
                    
                    if sources:
                        formatted_response += "**📚 Sources:**\n"
                        for url, title in list(sources.items())[:5]:
                            formatted_response += f"- [{title}]({url})\n"
                    
                    st.markdown(formatted_response)
                    
                    # Show retrieved context in expander
                    with st.expander("📄 View Retrieved Context"):
                        for i, context in enumerate(contexts[:3]):
                            st.write(f"**Context {i+1}:**")
                            st.write(context[:500] + "..." if len(context) > 500 else context)
                            st.write("---")
                    
                else:
                    response = "I couldn't find relevant information in the indexed content. Please make sure the website has been properly crawled."
                    st.write(response)
                
                st.session_state.messages.append({"role": "assistant", "content": formatted_response if docs else response})

st.markdown("---")
st.markdown("💡 **Tips:** This enhanced version provides comprehensive, detailed answers by using better context retrieval and prompting strategies.")