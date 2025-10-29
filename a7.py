import streamlit as st
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from typing import Optional, List, Dict, Any, Set
import urllib3
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from pydantic import Field
import re
import time
from urllib.parse import urljoin, urlparse
from collections import deque
from datetime import datetime
import concurrent.futures
from threading import Lock
import asyncio
import aiohttp
from functools import lru_cache

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FastWebCrawler:
    """Optimized web crawler with concurrent crawling"""
    
    def __init__(self, max_depth=2, max_pages=50, max_workers=5, use_selenium_threshold=0.3):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.max_workers = max_workers
        self.use_selenium_threshold = use_selenium_threshold  # Percentage of pages to use Selenium
        self.visited_urls = set()
        self.visited_lock = Lock()
        self.scraped_data = []
        self.data_lock = Lock()
        self.driver_pool = []
        self.driver_lock = Lock()
        self.session = None
        
    def setup_session(self):
        """Setup aiohttp session for faster requests"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=10)
            connector = aiohttp.TCPConnector(limit=20, force_close=True)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
    
    def get_chrome_options(self):
        """Get optimized Chrome options"""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-images")  # Don't load images
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_argument("--log-level=3")
        
        # Page load strategy
        options.page_load_strategy = 'eager'  # Don't wait for all resources
        
        # Disable loading of certain resources
        prefs = {
            "profile.default_content_setting_values": {
                "images": 2,
                "plugins": 2,
                "popups": 2,
                "geolocation": 2,
                "notifications": 2,
                "media_stream": 2,
                "media_stream_mic": 2,
                "media_stream_camera": 2,
                "protocol_handlers": 2,
                "ppapi_broker": 2,
                "automatic_downloads": 2,
                "midi_sysex": 2,
                "push_messaging": 2,
                "ssl_cert_decisions": 2,
                "metro_switch_to_desktop": 2,
                "protected_media_identifier": 2,
                "app_banner": 2,
                "site_engagement": 2,
                "durable_storage": 2
            }
        }
        options.add_experimental_option("prefs", prefs)
        
        return options
    
    def create_driver_pool(self, size=3):
        """Create a pool of drivers for reuse"""
        service = Service(ChromeDriverManager().install())
        options = self.get_chrome_options()
        
        for _ in range(size):
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(10)
            driver.implicitly_wait(2)
            self.driver_pool.append(driver)
    
    def get_driver(self):
        """Get a driver from the pool"""
        with self.driver_lock:
            if self.driver_pool:
                return self.driver_pool.pop()
            else:
                # Create a new one if pool is empty
                service = Service(ChromeDriverManager().install())
                options = self.get_chrome_options()
                driver = webdriver.Chrome(service=service, options=options)
                driver.set_page_load_timeout(10)
                return driver
    
    def return_driver(self, driver):
        """Return a driver to the pool"""
        with self.driver_lock:
            if len(self.driver_pool) < 3:
                self.driver_pool.append(driver)
            else:
                driver.quit()
    
    @lru_cache(maxsize=1000)
    def normalize_url(self, url):
        """Normalize and cache URL normalization"""
        url = url.split('#')[0].rstrip('/')
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    
    def is_valid_url(self, url, base_domain):
        """Quick URL validation"""
        try:
            parsed = urlparse(url)
            if parsed.netloc != urlparse(base_domain).netloc:
                return False
            
            # Quick pattern check
            invalid_patterns = [
                r'\.(jpg|jpeg|png|gif|pdf|zip|exe|css|js)$',
                r'(login|signin|register|logout)',
                r'#',
                r'javascript:',
                r'mailto:'
            ]
            
            url_lower = url.lower()
            for pattern in invalid_patterns:
                if re.search(pattern, url_lower):
                    return False
            
            return True
        except:
            return False
    
    async def extract_content_async(self, url):
        """Async content extraction using aiohttp"""
        try:
            async with self.session.get(url, ssl=False) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'lxml')  # lxml is faster
                
                # Quick extraction
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ""
                
                # Remove scripts and styles
                for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
                    tag.decompose()
                
                # Get main content
                main_content = ""
                for selector in ['main', 'article', '.content', '#content']:
                    element = soup.select_one(selector)
                    if element:
                        main_content = element.get_text(separator="\n", strip=True)
                        break
                
                if not main_content:
                    main_content = soup.get_text(separator="\n", strip=True)
                
                # Extract links
                links = set()
                for a in soup.find_all('a', href=True):
                    link = urljoin(url, a['href'])
                    normalized = self.normalize_url(link)
                    if self.is_valid_url(normalized, url):
                        links.add(normalized)
                
                # Clean content
                lines = [line.strip() for line in main_content.splitlines() 
                        if line.strip() and len(line.strip()) > 20]
                content = '\n'.join(lines[:500])  # Limit content size
                
                return {
                    'url': url,
                    'title': title_text[:200],
                    'content': content,
                    'links': list(links)[:50],  # Limit links
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return None
    
    def extract_content_selenium(self, url):
        """Selenium extraction for dynamic content"""
        driver = None
        try:
            driver = self.get_driver()
            driver.get(url)
            
            # Quick wait for main content
            time.sleep(1)
            
            # Get page source
            soup = BeautifulSoup(driver.page_source, 'lxml')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
                tag.decompose()
            
            # Get content
            content = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in content.splitlines() 
                    if line.strip() and len(line.strip()) > 20]
            content = '\n'.join(lines[:500])
            
            # Extract links
            links = set()
            for a in soup.find_all('a', href=True):
                link = urljoin(url, a['href'])
                normalized = self.normalize_url(link)
                if self.is_valid_url(normalized, url):
                    links.add(normalized)
            
            return {
                'url': url,
                'title': title_text[:200],
                'content': content,
                'links': list(links)[:50],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return None
        finally:
            if driver:
                self.return_driver(driver)
    
    async def crawl_url_async(self, url, depth, use_selenium=False):
        """Crawl a single URL asynchronously"""
        # Check if already visited
        with self.visited_lock:
            if url in self.visited_urls or len(self.scraped_data) >= self.max_pages:
                return []
            self.visited_urls.add(url)
        
        # Extract content
        if use_selenium:
            data = await asyncio.get_event_loop().run_in_executor(
                None, self.extract_content_selenium, url
            )
        else:
            data = await self.extract_content_async(url)
        
        if data and data['content']:
            with self.data_lock:
                self.scraped_data.append(data)
            
            # Return links for further crawling
            if depth < self.max_depth:
                return [(link, depth + 1) for link in data['links']]
        
        return []
    
    async def crawl_website_async(self, start_url, progress_callback=None):
        """Main async crawling function"""
        self.setup_session()
        
        # Create driver pool for Selenium
        if self.use_selenium_threshold > 0:
            self.create_driver_pool(min(3, self.max_workers))
        
        # Initialize queue
        queue = deque([(start_url, 0)])
        processed = 0
        
        try:
            while queue and len(self.scraped_data) < self.max_pages:
                # Get batch of URLs to process
                batch = []
                for _ in range(min(self.max_workers, len(queue))):
                    if queue:
                        batch.append(queue.popleft())
                
                if not batch:
                    break
                
                # Decide which URLs need Selenium (e.g., every 3rd URL)
                tasks = []
                for i, (url, depth) in enumerate(batch):
                    use_selenium = (i % int(1/self.use_selenium_threshold)) == 0 if self.use_selenium_threshold > 0 else False
                    tasks.append(self.crawl_url_async(url, depth, use_selenium))
                
                # Process batch concurrently
                results = await asyncio.gather(*tasks)
                
                # Add new URLs to queue
                for new_urls in results:
                    queue.extend(new_urls)
                
                # Update progress
                processed += len(batch)
                if progress_callback:
                    progress_callback(
                        len(self.scraped_data), 
                        len(self.visited_urls), 
                        batch[-1][0] if batch else ""
                    )
                
                # Small delay to be polite
                await asyncio.sleep(0.1)
            
        finally:
            # Cleanup
            if self.session:
                await self.session.close()
            
            # Close all drivers
            with self.driver_lock:
                for driver in self.driver_pool:
                    driver.quit()
                self.driver_pool.clear()
        
        return self.scraped_data

# Simplified UI components for faster processing
class FastEmbeddings(Embeddings):
    """Fast embeddings using smaller model"""
    
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
        # Process in batches for speed
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # Truncate texts
            batch_texts = [text[:512] for text in batch_texts]
            
            encoded = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(DEVICE)
            
            model_output = self.model(**encoded)
            embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().numpy().tolist())
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class FastLLM(LLM):
    """Optimized LLM for faster responses"""
    
    model_name: str = Field(default="google/flan-t5-base")
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(DEVICE)
        self.model.eval()
    
    @property
    def _llm_type(self) -> str:
        return "fast_llm"
    
    @torch.no_grad()
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(DEVICE)
        
        outputs = self.model.generate(
            **inputs,
            max_length=300,
            min_length=30,
            num_beams=3,  # Reduced beams for speed
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            early_stopping=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response if response else "I couldn't generate a response. Please try rephrasing your question."

# Streamlit UI
st.set_page_config(page_title="Fast Web Crawler & Q&A Bot", layout="wide")

st.title("⚡ Fast Web Crawler & Q&A Bot")
st.markdown("Optimized crawler for faster website indexing")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Speed Settings")
    crawl_mode = st.radio(
        "Crawl Mode",
        ["Fast (Mostly Static)", "Balanced", "Thorough (More Dynamic)"],
        index=0
    )
    
    if crawl_mode == "Fast (Mostly Static)":
        max_workers = 8
        use_selenium_threshold = 0.1
        max_depth = 2
        max_pages = 30
    elif crawl_mode == "Balanced":
        max_workers = 5
        use_selenium_threshold = 0.3
        max_depth = 2
        max_pages = 50
    else:
        max_workers = 3
        use_selenium_threshold = 0.5
        max_depth = 3
        max_pages = 100
    
    st.info(f"Workers: {max_workers}, Selenium: {int(use_selenium_threshold*100)}%, Depth: {max_depth}")
    
    # Manual overrides
    with st.expander("Advanced Settings"):
        max_depth = st.slider("Max depth", 1, 4, max_depth)
        max_pages = st.slider("Max pages", 10, 200, max_pages)
        max_workers = st.slider("Concurrent workers", 1, 10, max_workers)
    
    st.subheader("Model Settings")
    model_size = st.selectbox(
        "Model Size",
        ["Small (Fast)", "Base (Balanced)", "Large (Quality)"],
        index=1
    )
    
    if model_size == "Small (Fast)":
        model_name = "google/flan-t5-small"
    elif model_size == "Base (Balanced)":
        model_name = "google/flan-t5-base"
    else:
        model_name = "google/flan-t5-large"

# Initialize models
@st.cache_resource
def init_models(model_name):
    llm = FastLLM(model_name=model_name)
    embeddings = FastEmbeddings()
    return llm, embeddings

# Main interface
tab1, tab2 = st.tabs(["🚀 Fast Crawler", "💬 Q&A Bot"])

with tab1:
    st.header("Fast Web Crawler")
    
    url_input = st.text_input("Enter website URL:", placeholder="https://example.com")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        crawl_button = st.button("🚀 Start Fast Crawl", type="primary", use_container_width=True)
    with col2:
        estimate = max_pages * 0.5  # Rough estimate seconds
        st.metric("Est. Time", f"{estimate:.0f}s")
    with col3:
        if st.button("🗑️ Clear", use_container_width=True):
            if os.path.exists("fast_index"):
                import shutil
                shutil.rmtree("fast_index")
            st.rerun()
    
    if crawl_button and url_input:
        start_time = time.time()
        
        # Initialize crawler
        crawler = FastWebCrawler(
            max_depth=max_depth,
            max_pages=max_pages,
            max_workers=max_workers,
            use_selenium_threshold=use_selenium_threshold
        )
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.container()
        
        def update_progress(pages_crawled, total_visited, current_url):
            progress = pages_crawled / max_pages
            progress_bar.progress(min(progress, 1.0))
            elapsed = time.time() - start_time
            rate = pages_crawled / elapsed if elapsed > 0 else 0
            status_text.text(f"Pages: {pages_crawled}/{max_pages} | Rate: {rate:.1f} pages/sec | Current: {current_url[:60]}...")
        
        # Run async crawler
        async def run_crawler():
            return await crawler.crawl_website_async(url_input, progress_callback=update_progress)
        
        # Execute crawler
        with st.spinner("🏃 Crawling at high speed..."):
            scraped_data = asyncio.run(run_crawler())
        
        elapsed_time = time.time() - start_time
        
        if scraped_data:
            st.success(f"✅ Crawled {len(scraped_data)} pages in {elapsed_time:.1f} seconds!")
            
            # Show metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pages/sec", f"{len(scraped_data)/elapsed_time:.1f}")
            with col2:
                st.metric("Total Pages", len(scraped_data))
            with col3:
                st.metric("Unique URLs", len(crawler.visited_urls))
            with col4:
                st.metric("Time Saved", f"{(estimate-elapsed_time):.0f}s")
            
            # Process and index
            with st.spinner("⚡ Fast indexing..."):
                llm, embeddings = init_models(model_name)
                
                # Prepare texts
                texts = []
                metadata = []
                
                # Faster chunking
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100,
                    length_function=len
                )
                
                for page in scraped_data:
                    content = f"{page['title']}\n{page['content']}"
                    chunks = splitter.split_text(content)
                    
                    for chunk in chunks[:5]:  # Limit chunks per page
                        if len(chunk) > 50:
                            texts.append(chunk)
                            metadata.append({
                                'url': page['url'],
                                'title': page['title'][:100]
                            })
                
                # Create index
                vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
                vectorstore.save_local("fast_index")
                
                # Store in session
                st.session_state.vectorstore = vectorstore
                st.session_state.website_data = {
                    'total_pages': len(scraped_data),
                    'total_chunks': len(texts),
                    'domain': urlparse(url_input).netloc,
                    'crawl_time': elapsed_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.success(f"✅ Indexed {len(texts)} chunks in {time.time() - start_time - elapsed_time:.1f}s")

with tab2:
    st.header("Q&A Bot")
    
    # Load index if exists
    if "vectorstore" not in st.session_state:
        if os.path.exists("fast_index"):
            with st.spinner("Loading index..."):
                llm, embeddings = init_models(model_name)
                st.session_state.vectorstore = FAISS.load_local(
                    "fast_index", 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                st.success("✅ Index loaded!")
        else:
            st.warning("No index found. Please crawl a website first!")
            st.stop()
    
    # Show info
    if "website_data" in st.session_state:
        data = st.session_state.website_data
        st.info(f"📊 {data['total_pages']} pages | {data['total_chunks']} chunks | Crawled in {data['crawl_time']:.1f}s")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Fast retrieval
                docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Generate response
                llm_prompt = f"Context:\n{context[:2000]}\n\nQuestion: {prompt}\n\nAnswer:"
                
                if "llm" not in st.session_state:
                    st.session_state.llm = FastLLM(model_name=model_name)
                
                response = st.session_state.llm._call(llm_prompt)
                st.write(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("⚡ Optimized for speed: Concurrent crawling, smart content extraction, and efficient indexing")