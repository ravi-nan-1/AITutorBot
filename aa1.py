import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
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
import json

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
            
            # Extract structured content
            title = soup.find('title')
            title_text = title.get_text() if title else "No Title"
            
            # Try to extract main content
            main_content = None
            for selector in ['main', 'article', '.content', '#content', '[role="main"]']:
                element = soup.select_one(selector)
                if element:
                    main_content = element.get_text(separator='\n', strip=True)
                    break
            
            if not main_content:
                main_content = soup.get_text(separator='\n', strip=True)
            
            # Extract headings for structure
            headings = []
            for h in soup.find_all(['h1', 'h2', 'h3']):
                headings.append(h.get_text().strip())
            
            # Clean and limit text
            main_content = ' '.join(main_content.split())[:15000]
            
            return {
                'url': url,
                'title': title_text[:200],
                'content': main_content,
                'headings': headings[:10],
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
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(url, link['href'])
                        if base_domain in next_url and next_url not in self.visited_urls:
                            urls_to_visit.append(next_url)
                except:
                    pass
            
            time.sleep(0.3)  # Rate limiting
            
        return self.scraped_data

# ========== EMBEDDINGS ==========
class TutorEmbeddings(Embeddings):
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
        self.model = AutoModel.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
        self.model.eval()
    
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

# ========== TUTOR LLM ==========
class TutorLLM(LLM):
    model_name: str = Field(default="google/flan-t5-base")
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
        self.model.eval()
    
    @property
    def _llm_type(self) -> str:
        return "tutor_llm"
    
    @torch.no_grad()
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt[:1024], return_tensors="pt", 
                               max_length=1024, truncation=True)
        
        outputs = self.model.generate(
            **inputs,
            max_length=256,
            min_length=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        gc.collect()
        return response

# ========== RAG TUTOR SYSTEM ==========
class RAGTutor:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Different prompt templates for different tutoring modes
        self.prompts = {
            "explain": PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template="""You are an expert tutor. Use the following context to provide a clear, educational explanation.

Context: {context}

Previous conversation: {chat_history}

Student's question: {question}

Provide a detailed educational explanation that:
1. Starts with a simple overview
2. Explains key concepts clearly
3. Uses examples when helpful
4. Builds on previous knowledge
5. Checks understanding

Tutor's explanation:"""
            ),
            
            "summarize": PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template="""You are a tutor creating a study summary. Based on the context, create a concise summary.

Context: {context}

Student request: {question}

Create a structured summary with:
1. Main topics
2. Key points
3. Important facts
4. Connections between ideas

Summary:"""
            ),
            
            "quiz": PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template="""You are a tutor creating quiz questions. Based on the context, generate educational questions.

Context: {context}

Topic: {question}

Create 3 questions that test understanding:
1. One factual question
2. One conceptual question  
3. One application question

Include brief answers for each.

Quiz questions:"""
            ),
            
            "simplify": PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template="""You are a tutor explaining to a beginner. Simplify this concept.

Context: {context}

Concept to simplify: {question}

Explain this in the simplest terms possible:
- Use everyday language
- Give relatable examples
- Avoid jargon
- Use analogies

Simple explanation:"""
            ),
            
            "elaborate": PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template="""You are an advanced tutor providing deep insights.

Context: {context}

Previous discussion: {chat_history}

Topic to elaborate on: {question}

Provide an in-depth analysis including:
1. Detailed explanation
2. Historical context or background
3. Different perspectives
4. Advanced concepts
5. Real-world applications

Detailed elaboration:"""
            )
        }
        
        # Create retrieval chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )
    
    def get_response(self, question: str, mode: str = "explain", difficulty: str = "medium"):
        """Get tutoring response based on mode and difficulty"""
        
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=5)
        
        if not docs:
            return {
                "answer": "I don't have enough information about this topic. Could you please provide more context or rephrase your question?",
                "sources": [],
                "suggestions": []
            }
        
        # Prepare context
        context = "\n\n".join([doc.page_content for doc in docs[:4]])
        
        # Get chat history
        chat_history = self.memory.buffer_as_str if hasattr(self.memory, 'buffer_as_str') else ""
        
        # Adjust prompt based on difficulty
        difficulty_prefix = {
            "beginner": "Explain simply as if to someone new to this topic: ",
            "medium": "",
            "advanced": "Provide an advanced, detailed explanation: "
        }
        
        # Format prompt
        prompt_template = self.prompts.get(mode, self.prompts["explain"])
        prompt = prompt_template.format(
            context=context[:2000],
            question=difficulty_prefix.get(difficulty, "") + question,
            chat_history=chat_history[-500:] if chat_history else "None"
        )
        
        # Get response
        try:
            response = self.llm._call(prompt)
            
            # Store in memory
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(response)
            
            # Generate follow-up suggestions
            suggestions = self.generate_suggestions(question, response, docs)
            
            return {
                "answer": response,
                "sources": [{"title": doc.metadata.get("title", ""), 
                           "url": doc.metadata.get("url", "")} for doc in docs[:3]],
                "suggestions": suggestions
            }
        except Exception as e:
            return {
                "answer": f"I encountered an error: {str(e)}. Please try rephrasing your question.",
                "sources": [],
                "suggestions": []
            }
    
    def generate_suggestions(self, question: str, response: str, docs: List):
        """Generate follow-up learning suggestions"""
        suggestions = []
        
        # Extract key topics from response
        topics = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response)[:3]
        
        if topics:
            suggestions.append(f"Tell me more about {topics[0]}")
        
        suggestions.extend([
            "Can you give me an example?",
            "How does this relate to real-world applications?",
            "Can you quiz me on this topic?",
            "Explain this more simply"
        ])
        
        return suggestions[:4]
    
    def assess_understanding(self, question: str, answer: str):
        """Assess student's answer to a question"""
        prompt = f"""As a tutor, evaluate this answer:
        Question: {question}
        Student's answer: {answer}
        
        Provide:
        1. Whether the answer is correct
        2. What's good about the answer
        3. What could be improved
        4. The correct answer if wrong
        
        Evaluation:"""
        
        return self.llm._call(prompt)

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="AI Tutor - RAG Learning Assistant", layout="wide", page_icon="🎓")

# Custom CSS for better UI
st.markdown("""
<style>
    .tutor-mode-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .source-box {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 AI Tutor - RAG-Powered Learning Assistant")
st.markdown("Your personal AI tutor that learns from any website to help you understand complex topics!")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Tutor Settings")
    
    # Learning Mode
    st.subheader("🎯 Learning Mode")
    learning_mode = st.selectbox(
        "Choose your learning style:",
        ["Explain", "Simplify", "Elaborate", "Summarize", "Quiz"],
        help="Different ways the tutor can help you learn"
    )
    
    # Difficulty Level
    difficulty = st.selectbox(
        "📊 Difficulty Level:",
        ["Beginner", "Medium", "Advanced"],
        index=1
    )
    
    # Study Features
    st.subheader("📚 Study Features")
    show_sources = st.checkbox("Show source references", value=True)
    show_suggestions = st.checkbox("Show follow-up suggestions", value=True)
    track_progress = st.checkbox("Track learning progress", value=False)
    
    # Crawl Settings
    st.subheader("🕸️ Content Settings")
    max_pages = st.slider("Pages to analyze", 5, 25, 10)
    
    # Clear Data
    if st.button("🗑️ Clear All Data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("All data cleared!")
        st.rerun()

# Main Content Area
tab1, tab2, tab3, tab4 = st.tabs(["📖 Load Content", "💬 Learn with Tutor", "📝 Practice", "📊 Progress"])

# ========== TAB 1: CONTENT LOADING ==========
with tab1:
    st.header("📖 Load Learning Content")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        url_input = st.text_input(
            "Enter a URL to learn from:",
            placeholder="https://en.wikipedia.org/wiki/Machine_learning",
            help="Enter any educational website, Wikipedia article, or documentation"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")
        load_button = st.button("🔍 Load Content", type="primary", use_container_width=True)
    
    # Example URLs
    st.markdown("**Quick Start Examples:**")
    example_cols = st.columns(4)
    examples = [
        ("🧬 Biology", "https://en.wikipedia.org/wiki/DNA"),
        ("🔬 Physics", "https://en.wikipedia.org/wiki/Quantum_mechanics"),
        ("💻 Computer Science", "https://en.wikipedia.org/wiki/Artificial_intelligence"),
        ("📐 Mathematics", "https://en.wikipedia.org/wiki/Calculus")
    ]
    
    for col, (label, url) in zip(example_cols, examples):
        with col:
            if st.button(label, use_container_width=True):
                url_input = url
                load_button = True
    
    if load_button and url_input:
        with st.spinner("📚 Loading and analyzing content..."):
            # Progress tracking
            progress = st.progress(0)
            status = st.empty()
            
            # Crawl website
            status.text("🕸️ Crawling website...")
            crawler = TutorWebCrawler(max_pages=max_pages)
            scraped_data = crawler.crawl_website(url_input)
            progress.progress(0.3)
            
            if scraped_data:
                status.text(f"📄 Processing {len(scraped_data)} pages...")
                
                # Create embeddings
                embeddings = TutorEmbeddings()
                progress.progress(0.5)
                
                # Process content for RAG
                texts = []
                metadata = []
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ". ", "! ", "? ", ", "]
                )
                
                for page in scraped_data:
                    # Add headings context
                    headings_context = " | ".join(page.get('headings', []))
                    enhanced_content = f"Title: {page['title']}\nTopics: {headings_context}\n\n{page['content']}"
                    
                    chunks = splitter.split_text(enhanced_content)
                    for i, chunk in enumerate(chunks[:8]):  # Limit chunks per page
                        texts.append(chunk)
                        metadata.append({
                            'url': page['url'],
                            'title': page['title'],
                            'chunk_id': i
                        })
                
                progress.progress(0.7)
                status.text("🧠 Creating knowledge base...")
                
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
                    'chunks': len(texts),
                    'topics': list(set([page['title'] for page in scraped_data]))[:5]
                }
                
                progress.progress(1.0)
                status.empty()
                
                # Success message
                st.success(f"""✅ Content loaded successfully!
                - 📄 Analyzed {len(scraped_data)} pages
                - 🧩 Created {len(texts)} knowledge chunks
                - 🎓 Ready to start learning!""")
                
                # Show content overview
                with st.expander("📋 Content Overview"):
                    for page in scraped_data[:5]:
                        st.write(f"• **{page['title']}**")
                        if page.get('headings'):
                            st.write(f"  Topics: {', '.join(page['headings'][:3])}")
            else:
                st.error("Could not load content from the URL. Please try another.")

# ========== TAB 2: TUTORING ==========
with tab2:
    st.header("💬 Learn with Your AI Tutor")
    
    if "content_loaded" not in st.session_state:
        st.info("📚 Please load content from a website first (go to 'Load Content' tab)")
    else:
        # Show loaded content info
        info = st.session_state.content_info
        st.success(f"📖 Learning from: {info['url']}")
        st.caption(f"Knowledge base: {info['chunks']} chunks from {info['pages']} pages")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Hello! I'm your AI tutor. I've studied the content from {info['url']}. What would you like to learn about today?"
            })
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources if available
                if message.get("sources") and show_sources:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"• [{source['title']}]({source['url']})")
        
        # Suggestion buttons
        if show_suggestions and st.session_state.messages:
            last_message = st.session_state.messages[-1]
            if last_message.get("suggestions"):
                st.write("**🤔 Suggested questions:**")
                cols = st.columns(len(last_message["suggestions"]))
                for col, suggestion in zip(cols, last_message["suggestions"]):
                    with col:
                        if st.button(suggestion, key=f"sug_{suggestion[:20]}"):
                            st.session_state.next_question = suggestion
        
        # Chat input
        if "next_question" in st.session_state:
            prompt = st.session_state.next_question
            del st.session_state.next_question
        else:
            prompt = st.chat_input("Ask me anything about the loaded content...")
        
        if prompt:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get tutor response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    tutor = st.session_state.tutor
                    response = tutor.get_response(
                        prompt, 
                        mode=learning_mode.lower(),
                        difficulty=difficulty.lower()
                    )
                    
                    st.markdown(response["answer"])
                    
                    # Store message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                        "suggestions": response["suggestions"]
                    })
                    
                    # Track progress if enabled
                    if track_progress:
                        if "learning_history" not in st.session_state:
                            st.session_state.learning_history = []
                        st.session_state.learning_history.append({
                            "question": prompt,
                            "mode": learning_mode,
                            "timestamp": datetime.now().isoformat()
                        })

# ========== TAB 3: PRACTICE ==========
with tab3:
    st.header("📝 Practice & Assessment")
    
    if "content_loaded" not in st.session_state:
        st.info("📚 Please load content first")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Quick Quiz")
            if st.button("Generate Quiz Questions"):
                with st.spinner("Creating quiz..."):
                    tutor = st.session_state.tutor
                    quiz_topic = st.text_input("Quiz topic (leave empty for general):", "")
                    
                    response = tutor.get_response(
                        quiz_topic or "Create a quiz about the main topics",
                        mode="quiz"
                    )
                    
                    st.markdown("### Quiz Questions:")
                    st.markdown(response["answer"])
        
        with col2:
            st.subheader("✍️ Test Your Understanding")
            test_question = st.text_area("Question to answer:")
            your_answer = st.text_area("Your answer:")
            
            if st.button("Check Answer"):
                if test_question and your_answer:
                    with st.spinner("Evaluating..."):
                        evaluation = st.session_state.tutor.assess_understanding(
                            test_question, your_answer
                        )
                        st.markdown("### Feedback:")
                        st.markdown(evaluation)

# ========== TAB 4: PROGRESS ==========
with tab4:
    st.header("📊 Learning Progress")
    
    if track_progress and "learning_history" in st.session_state:
        st.subheader("📈 Your Learning Journey")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", len(st.session_state.learning_history))
        with col2:
            modes_used = set([h["mode"] for h in st.session_state.learning_history])
            st.metric("Learning Modes Used", len(modes_used))
        with col3:
            st.metric("Active Sessions", len(st.session_state.messages) // 2)
        
        # History
        st.subheader("📝 Recent Questions")
        for item in st.session_state.learning_history[-5:]:
            st.write(f"• {item['question']} ({item['mode']}) - {item['timestamp'][:10]}")
    else:
        st.info("Enable 'Track learning progress' in settings to see your progress")

# Footer
st.markdown("---")
st.markdown("🎓 **AI Tutor** - Powered by RAG (Retrieval-Augmented Generation)")
st.caption("Learn anything from any website with your personal AI tutor!")