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
                               '.infobox', '.hatnote', '.reference', '.reflist']:
                    for element in soup.select(selector):
                        element.decompose()
                
                # Get main content
                content_div = soup.find('div', {'id': 'mw-content-text'})
                if content_div:
                    content = content_div.get_text(separator='\n', strip=True)
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

# Simple, fast embeddings
class FastEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
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
        for text in texts:
            text = text[:512]  # Truncate
            encoded = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
            model_output = self.model(**encoded)
            pooled = self.mean_pooling(model_output, encoded['attention_mask'])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(normalized.cpu().numpy()[0].tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class SimpleLLM(LLM):
    model_name: str = Field(default="google/flan-t5-base")
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(DEVICE)
    
    @property
    def _llm_type(self) -> str:
        return "simple_llm"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
        outputs = self.model.generate(**inputs, max_length=300, num_beams=3, temperature=0.7)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.set_page_config(page_title="Robust Web Crawler & Q&A", layout="wide")
st.title("🔍 Robust Web Crawler & Q&A Bot")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    max_pages = st.slider("Max pages to crawl", 10, 100, 30)
    max_depth = st.slider("Max crawl depth", 1, 3, 2)
    timeout = st.slider("Request timeout (seconds)", 5, 30, 15)

# Main tabs
tab1, tab2 = st.tabs(["🕸️ Crawler", "💬 Q&A"])

with tab1:
    url_input = st.text_input("Enter URL to crawl:", placeholder="https://en.wikipedia.org/wiki/Python_(programming_language)")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        crawl_button = st.button("Start Crawling", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear Data", use_container_width=True):
            if os.path.exists("robust_index"):
                import shutil
                shutil.rmtree("robust_index")
            if "vectorstore" in st.session_state:
                del st.session_state.vectorstore
            st.rerun()
    
    if crawl_button and url_input:
        start_time = time.time()
        
        # Initialize crawler
        crawler = RobustWebCrawler(max_depth=max_depth, max_pages=max_pages, timeout=timeout)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        error_text = st.empty()
        
        def update_progress(pages_crawled, total_visited, current_url):
            progress = pages_crawled / max_pages if max_pages > 0 else 0
            progress_bar.progress(min(progress, 1.0))
            
            elapsed = time.time() - start_time
            rate = pages_crawled / elapsed if elapsed > 0 else 0
            
            status_text.text(f"✅ Crawled: {pages_crawled} | ⏳ Visited: {total_visited} | 📊 Rate: {rate:.1f} pages/sec")
            
            if crawler.failed_urls:
                error_text.warning(f"⚠️ Failed URLs: {len(crawler.failed_urls)}")
        
        # Crawl website
        with st.spinner("🔍 Crawling website..."):
            scraped_data = crawler.crawl_website(url_input, progress_callback=update_progress)
        
        elapsed_time = time.time() - start_time
        
        if scraped_data:
            st.success(f"✅ Successfully crawled {len(scraped_data)} pages in {elapsed_time:.1f} seconds!")
            
            # Show statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pages Crawled", len(scraped_data))
            with col2:
                st.metric("Total Visited", len(crawler.visited_urls))
            with col3:
                st.metric("Failed URLs", len(crawler.failed_urls))
            with col4:
                st.metric("Avg Time/Page", f"{elapsed_time/len(scraped_data):.1f}s")
            
            # Show sample content
            with st.expander("📄 Sample Crawled Content"):
                for i, page in enumerate(scraped_data[:3]):
                    st.write(f"**{i+1}. {page['title']}**")
                    st.write(f"URL: {page['url']}")
                    st.write(f"Content preview: {page['content'][:200]}...")
                    st.write("---")
            
            # Index content
            with st.spinner("📚 Indexing content..."):
                # Initialize models
                embeddings = FastEmbeddings()
                
                # Prepare texts
                texts = []
                metadata = []
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100
                )
                
                for page in scraped_data:
                    content = f"Title: {page['title']}\n\n{page['content']}"
                    chunks = splitter.split_text(content)
                    
                    for chunk in chunks[:10]:  # Limit chunks per page
                        if len(chunk) > 50:
                            texts.append(chunk)
                            metadata.append({
                                'url': page['url'],
                                'title': page['title']
                            })
                
                # Create and save index
                vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
                vectorstore.save_local("robust_index")
                
                st.session_state.vectorstore = vectorstore
                st.session_state.crawl_info = {
                    'pages': len(scraped_data),
                    'chunks': len(texts),
                    'time': elapsed_time,
                    'domain': urlparse(url_input).netloc
                }
                
                st.success(f"✅ Indexed {len(texts)} text chunks!")
        else:
            st.error("❌ No data was crawled. Please check the URL and try again.")
            if crawler.failed_urls:
                with st.expander("Failed URLs"):
                    for url in list(crawler.failed_urls)[:10]:
                        st.write(f"- {url}")

with tab2:
    st.header("Q&A Bot")
    
    # Load index if exists
    if "vectorstore" not in st.session_state:
        if os.path.exists("robust_index"):
            with st.spinner("Loading index..."):
                embeddings = FastEmbeddings()
                st.session_state.vectorstore = FAISS.load_local(
                    "robust_index",
                    embeddings,
                    allow_dangerous_deserialization=True
                )
        else:
            st.warning("No indexed content found. Please crawl a website first!")
            st.stop()
    
    # Show info
    if "crawl_info" in st.session_state:
        info = st.session_state.crawl_info
        st.info(f"📊 Indexed: {info['pages']} pages, {info['chunks']} chunks from {info['domain']}")
    
    # Initialize LLM
    if "llm" not in st.session_state:
        st.session_state.llm = SimpleLLM()
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("Ask a question about the crawled content..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                # Search for relevant content
                docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
                
                if docs:
                    context = "\n\n".join([doc.page_content for doc in docs])
                    sources = list(set([doc.metadata.get('url', '') for doc in docs if doc.metadata]))
                    
                    # Generate response
                    llm_prompt = f"Based on the following context, answer the question.\n\nContext:\n{context[:2000]}\n\nQuestion: {prompt}\n\nAnswer:"
                    
                    response = st.session_state.llm._call(llm_prompt)
                    
                    # Add sources
                    if sources:
                        response += "\n\n**Sources:**\n"
                        for source in sources[:3]:
                            response += f"- {source}\n"
                else:
                    response = "I couldn't find relevant information in the indexed content."
                
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.markdown("💡 **Tips:** This crawler handles Wikipedia and most websites. For dynamic sites, increase timeout.")