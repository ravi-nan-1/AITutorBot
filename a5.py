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
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from typing import Optional, List, Dict, Any, Set
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
from urllib.parse import urljoin, urlparse, parse_qs
from collections import deque
import hashlib
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WebCrawler:
    """Comprehensive web crawler that follows all links and extracts content"""
    
    def __init__(self, max_depth=3, max_pages=100, use_selenium=True):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.use_selenium = use_selenium
        self.visited_urls = set()
        self.url_queue = deque()
        self.scraped_data = []
        self.domain_cache = {}
        
        if use_selenium:
            self.setup_selenium()
    
    def setup_selenium(self):
        """Setup Selenium with Chrome driver"""
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.chrome_options.add_experimental_option('useAutomationExtension', False)
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
    
    def normalize_url(self, url):
        """Normalize URL to avoid duplicates"""
        # Remove fragment
        url = url.split('#')[0]
        # Remove trailing slash
        url = url.rstrip('/')
        # Remove common tracking parameters
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        filtered_params = {k: v for k, v in query_params.items() 
                         if k not in ['utm_source', 'utm_medium', 'utm_campaign', 'ref', 'source']}
        
        # Reconstruct URL without tracking params
        if filtered_params:
            query_string = '&'.join([f"{k}={v[0]}" for k, v in filtered_params.items()])
            url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{query_string}"
        else:
            url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        return url
    
    def is_valid_url(self, url, base_domain):
        """Check if URL should be crawled"""
        try:
            parsed = urlparse(url)
            base_parsed = urlparse(base_domain)
            
            # Check if same domain
            if parsed.netloc != base_parsed.netloc:
                return False
            
            # Avoid non-content URLs
            skip_patterns = [
                r'\.(jpg|jpeg|png|gif|pdf|zip|exe|dmg|iso|doc|docx|xls|xlsx)$',
                r'/tag/',
                r'/author/',
                r'/page/\d+',
                r'#',
                r'javascript:',
                r'mailto:',
                r'/wp-admin',
                r'/admin',
                r'/login',
                r'/signin',
                r'/signup',
                r'/register'
            ]
            
            for pattern in skip_patterns:
                if re.search(pattern, url.lower()):
                    return False
            
            return True
        except:
            return False
    
    def extract_links(self, soup, base_url):
        """Extract all valid links from a page"""
        links = set()
        
        for tag in soup.find_all(['a', 'link']):
            href = tag.get('href')
            if href:
                absolute_url = urljoin(base_url, href)
                normalized_url = self.normalize_url(absolute_url)
                
                if self.is_valid_url(normalized_url, base_url):
                    links.add(normalized_url)
        
        return links
    
    def extract_content_selenium(self, url):
        """Extract content using Selenium"""
        driver = None
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.chrome_options)
            driver.get(url)
            
            # Wait for content to load
            time.sleep(2)
            
            # Try to find and click "accept cookies" buttons if present
            try:
                cookie_buttons = driver.find_elements(By.XPATH, 
                    "//button[contains(text(), 'Accept') or contains(text(), 'OK') or contains(text(), 'Got it')]")
                for button in cookie_buttons[:1]:  # Click only the first one
                    button.click()
                    time.sleep(0.5)
            except:
                pass
            
            # Get page source
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                               'aside', 'iframe', 'noscript']):
                element.decompose()
            
            # Remove ads and popups
            for element in soup.find_all(class_=re.compile(r'(popup|modal|overlay|ad|ads|advertisement|banner|cookie|gdpr)', re.I)):
                element.decompose()
            
            # Extract main content
            content_text = ""
            
            # Try to find main content areas
            main_selectors = [
                'main',
                'article',
                '[role="main"]',
                '.main-content',
                '#main-content',
                '.content',
                '#content',
                '.post-content',
                '.entry-content',
                '.page-content',
                '.text-content',
                'div.content',
                # W3Schools specific
                '.w3-main',
                '#main',
                '#belowtopnav',
                '.w3-container'
            ]
            
            for selector in main_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        text = element.get_text(separator="\n", strip=True)
                        if len(text) > 100:  # Meaningful content
                            content_text += text + "\n\n"
            
            # If no main content found, get body text
            if not content_text:
                body = soup.find('body')
                if body:
                    content_text = body.get_text(separator="\n", strip=True)
            
            # Extract all links for crawling
            links = self.extract_links(soup, url)
            
            # Clean the content
            lines = []
            for line in content_text.splitlines():
                line = line.strip()
                if (line and 
                    len(line) > 10 and
                    not re.match(r'^(Cookie|Privacy|Terms|©|Copyright)', line, re.I)):
                    lines.append(line)
            
            final_content = '\n'.join(lines)
            
            return {
                'url': url,
                'title': title_text,
                'content': final_content,
                'links': links,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            st.error(f"Error crawling {url}: {str(e)}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def extract_content_requests(self, url):
        """Fallback extraction using requests"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Get text content
            content_text = soup.get_text(separator="\n", strip=True)
            
            # Extract links
            links = self.extract_links(soup, url)
            
            return {
                'url': url,
                'title': title_text,
                'content': content_text,
                'links': links,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            st.error(f"Error with requests for {url}: {str(e)}")
            return None
    
    def crawl_website(self, start_url, progress_callback=None):
        """Crawl entire website starting from the given URL"""
        # Normalize and add start URL
        start_url = self.normalize_url(start_url)
        self.url_queue.append((start_url, 0))
        base_domain = urlparse(start_url).netloc
        
        pages_crawled = 0
        
        while self.url_queue and pages_crawled < self.max_pages:
            current_url, depth = self.url_queue.popleft()
            
            # Skip if already visited or too deep
            if current_url in self.visited_urls or depth > self.max_depth:
                continue
            
            self.visited_urls.add(current_url)
            
            # Update progress
            if progress_callback:
                progress_callback(pages_crawled, len(self.visited_urls), current_url)
            
            # Extract content
            if self.use_selenium:
                data = self.extract_content_selenium(current_url)
            else:
                data = self.extract_content_requests(current_url)
            
            if data and data['content']:
                self.scraped_data.append(data)
                pages_crawled += 1
                
                # Add new links to queue
                if depth < self.max_depth:
                    for link in data['links']:
                        if link not in self.visited_urls:
                            self.url_queue.append((link, depth + 1))
            
            # Be polite to the server
            time.sleep(0.5)
        
        return self.scraped_data

class ProfessionalEmbeddings(Embeddings):
    """Enhanced embeddings for professional responses"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
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
                # Truncate very long texts
                text = text[:2000]
                encoded = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
                model_output = self.model(**encoded)
                sentence_embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                embeddings.append(sentence_embeddings.cpu().numpy()[0].tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class ProfessionalLLM(LLM):
    """LLM configured for professional chatbot responses"""
    
    model_name: str = Field(default="google/flan-t5-large")
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(DEVICE)
    
    @property
    def _llm_type(self) -> str:
        return "professional_llm"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=500,
                min_length=50,
                num_beams=5,
                temperature=0.7,
                do_sample=True,
                top_p=0.92,
                repetition_penalty=1.5,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Ensure professional tone
        if not response or len(response) < 20:
            response = "I apologize, but I couldn't find sufficient information to answer your question. Could you please rephrase or provide more context?"
        
        return response

def create_professional_prompt(context: str, question: str, website_info: dict = None) -> str:
    """Create a professional prompt for the chatbot"""
    
    prompt = f"""You are a professional assistant with expertise in the content from the crawled website. 
Provide a comprehensive, well-structured, and informative response to the user's question.

Website Information:
- Total pages indexed: {website_info.get('total_pages', 'Unknown')}
- Domain: {website_info.get('domain', 'Unknown')}
- Last updated: {website_info.get('last_updated', 'Unknown')}

Context from the website:
{context[:3000]}

User Question: {question}

Instructions:
1. Provide a detailed and accurate answer based on the context
2. Structure your response with clear sections if needed
3. Include specific examples or references from the content when relevant
4. Maintain a professional and helpful tone
5. If the information is not available in the context, acknowledge this professionally

Professional Response:"""
    
    return prompt

# Streamlit UI
st.set_page_config(page_title="Professional Web Crawler & Q&A Bot", layout="wide")

st.title("🤖 Professional Web Crawler & Q&A Bot")
st.markdown("Crawl entire websites and get professional answers to your questions")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Crawler Settings")
    max_depth = st.slider("Maximum crawl depth", 1, 5, 2)
    max_pages = st.slider("Maximum pages to crawl", 10, 500, 50)
    use_selenium = st.checkbox("Use Selenium (for dynamic sites)", value=True)
    
    st.subheader("Model Settings")
    model_options = {
        "google/flan-t5-large": "FLAN-T5 Large (Best quality, 780M params)",
        "google/flan-t5-base": "FLAN-T5 Base (Balanced, 220M params)",
        "google/flan-t5-small": "FLAN-T5 Small (Fast, 60M params)"
    }
    
    selected_model = st.selectbox(
        "Language Model",
        list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )
    
    embedding_model = st.selectbox(
        "Embedding Model",
        ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"]
    )

# Initialize models
@st.cache_resource
def init_models(model_name, embedding_model_name):
    llm = ProfessionalLLM(model_name=model_name)
    embeddings = ProfessionalEmbeddings(model_name=embedding_model_name)
    return llm, embeddings

# Main interface
tab1, tab2, tab3 = st.tabs(["🕸️ Web Crawler", "💬 Q&A Bot", "📊 Indexed Content"])

with tab1:
    st.header("Web Crawler")
    
    url_input = st.text_input("Enter website URL to crawl:", placeholder="https://example.com")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        crawl_button = st.button("🚀 Start Crawling", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear Index", use_container_width=True)
    
    if clear_button:
        if os.path.exists("website_index"):
            import shutil
            shutil.rmtree("website_index")
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        if "website_data" in st.session_state:
            del st.session_state.website_data
        st.success("Index cleared!")
    
    if crawl_button and url_input:
        # Initialize crawler
        crawler = WebCrawler(max_depth=max_depth, max_pages=max_pages, use_selenium=use_selenium)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(pages_crawled, total_visited, current_url):
            progress = pages_crawled / max_pages
            progress_bar.progress(progress)
            status_text.text(f"Crawling: {current_url[:80]}... ({pages_crawled}/{max_pages} pages)")
        
        # Start crawling
        with st.spinner("Initializing crawler..."):
            scraped_data = crawler.crawl_website(url_input, progress_callback=update_progress)
        
        if scraped_data:
            st.success(f"✅ Successfully crawled {len(scraped_data)} pages!")
            
            # Process and index the data
            with st.spinner("Processing and indexing content..."):
                # Initialize models
                llm, embeddings = init_models(selected_model, embedding_model)
                
                # Prepare texts for indexing
                texts = []
                metadata = []
                
                for page in scraped_data:
                    # Create chunks from each page
                    content = f"Title: {page['title']}\n\nContent: {page['content']}"
                    
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""]
                    )
                    
                    chunks = splitter.split_text(content)
                    
                    for chunk in chunks:
                        if len(chunk) > 50:
                            texts.append(chunk)
                            metadata.append({
                                'url': page['url'],
                                'title': page['title'],
                                'timestamp': page['timestamp']
                            })
                
                # Create vector store
                vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
                
                # Save to disk
                vectorstore.save_local("website_index")
                
                # Store in session state
                st.session_state.vectorstore = vectorstore
                st.session_state.website_data = {
                    'total_pages': len(scraped_data),
                    'total_chunks': len(texts),
                    'domain': urlparse(url_input).netloc,
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'pages': scraped_data
                }
                
                st.success(f"✅ Indexed {len(texts)} text chunks from {len(scraped_data)} pages!")
                
                # Show statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pages Crawled", len(scraped_data))
                with col2:
                    st.metric("Text Chunks", len(texts))
                with col3:
                    st.metric("Unique URLs", len(crawler.visited_urls))
                with col4:
                    st.metric("Crawl Depth", max_depth)

with tab2:
    st.header("Q&A Bot")
    
    # Check if index exists
    if "vectorstore" not in st.session_state:
        if os.path.exists("website_index"):
            # Load existing index
            with st.spinner("Loading indexed content..."):
                llm, embeddings = init_models(selected_model, embedding_model)
                st.session_state.vectorstore = FAISS.load_local(
                    "website_index", 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                st.success("✅ Loaded existing index!")
        else:
            st.warning("⚠️ No indexed content found. Please crawl a website first!")
            st.stop()
    
    # Website info
    if "website_data" in st.session_state:
        st.info(f"📊 Indexed: {st.session_state.website_data['total_pages']} pages from {st.session_state.website_data['domain']}")
    
    # Chat interface
    st.subheader("Ask me anything about the indexed content")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant content
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 6, "fetch_k": 12}
                )
                
                relevant_docs = retriever.get_relevant_documents(prompt)
                
                # Combine context with metadata
                context_parts = []
                urls_used = set()
                
                for doc in relevant_docs:
                    content = doc.page_content
                    if doc.metadata:
                        url = doc.metadata.get('url', '')
                        title = doc.metadata.get('title', '')
                        if url not in urls_used:
                            urls_used.add(url)
                            context_parts.append(f"[Source: {title}]\n{content}\n")
                        else:
                            context_parts.append(f"{content}\n")
                    else:
                        context_parts.append(f"{content}\n")
                
                context = "\n".join(context_parts)
                
                # Create professional prompt
                website_info = st.session_state.get('website_data', {})
                professional_prompt = create_professional_prompt(context, prompt, website_info)
                
                # Generate response
                if "llm" not in st.session_state:
                    st.session_state.llm = llm
                
                response = st.session_state.llm._call(professional_prompt)
                
                # Format response with sources
                formatted_response = f"{response}\n\n"
                if urls_used:
                    formatted_response += "**Sources:**\n"
                    for url in list(urls_used)[:3]:
                        formatted_response += f"- {url}\n"
                
                st.markdown(formatted_response)
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
    
    # Advanced options
    with st.expander("⚙️ Advanced Search Options"):
        col1, col2 = st.columns(2)
        with col1:
            search_type = st.selectbox("Search Type", ["mmr", "similarity"])
            num_results = st.slider("Number of results", 3, 10, 6)
        with col2:
            show_sources = st.checkbox("Always show sources", value=True)
            show_context = st.checkbox("Show retrieved context", value=False)

with tab3:
    st.header("Indexed Content")
    
    if "website_data" in st.session_state:
        st.subheader("Website Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pages", st.session_state.website_data['total_pages'])
        with col2:
            st.metric("Total Chunks", st.session_state.website_data.get('total_chunks', 'N/A'))
        with col3:
            st.metric("Last Updated", st.session_state.website_data['last_updated'])
        
        st.subheader("Crawled Pages")
        
        # Search within crawled pages
        search_query = st.text_input("Search in crawled pages:", placeholder="Enter keywords...")
        
        pages_to_show = st.session_state.website_data['pages']
        
        if search_query:
            # Filter pages based on search
            filtered_pages = []
            for page in pages_to_show:
                if (search_query.lower() in page['title'].lower() or 
                    search_query.lower() in page['content'].lower() or
                    search_query.lower() in page['url'].lower()):
                    filtered_pages.append(page)
            pages_to_show = filtered_pages
        
        # Display pages
        st.write(f"Showing {len(pages_to_show)} pages")
        
        for i, page in enumerate(pages_to_show[:20]):  # Show max 20 pages
            with st.expander(f"📄 {page['title'][:80]}..."):
                st.write(f"**URL:** {page['url']}")
                st.write(f"**Crawled:** {page['timestamp']}")
                st.write(f"**Content Preview:**")
                st.text(page['content'][:500] + "..." if len(page['content']) > 500 else page['content'])
                st.write(f"**Links found:** {len(page.get('links', []))}")
    else:
        st.info("No content indexed yet. Please crawl a website first!")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** For best results, crawl websites with well-structured content and clear navigation.")