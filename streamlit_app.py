import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import os
import time
import hashlib
from datetime import datetime
from dotenv import load_dotenv
import logging
import pandas as pd
import httpx

# Configure logging - console only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Chatbot Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .danger-button {
        background-color: #dc3545 !important;
        color: white !important;
        border: none !important;
    }
    
    .login-container {
        max-width: 400px;
        margin: 50px auto;
        padding: 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .login-header {
        text-align: center;
        color: #667eea;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def simple_auth():
    """Simple password-based authentication"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        # Center the login form
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        st.markdown('<h2 class="login-header">🔐 Authentication Required</h2>', unsafe_allow_html=True)
        st.markdown("Please enter the password to access the RAG Chatbot.")
        
        # Create a form for better UX
        with st.form("login_form"):
            password = st.text_input("Password:", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if submit_button and password:
                # Hash the entered password using MD5 (to match the secrets file)
                # hashed_password = hashlib.md5(password.encode()).hexdigest()
                
                # Get correct hash from secrets
                try:
                    correct_hash = st.secrets.get("APP_PASSWORD_HASH")
                    correct_pass = st.secrets.get("APP_PASSWORD")
                    # if not correct_hash:
                    #     st.error("❌ Authentication not configured properly. Contact administrator.")
                    #     st.stop()

                    if password == correct_pass:
                        st.session_state.authenticated = True
                        st.success("✅ Login successful! Redirecting...")
                        time.sleep(1)
                        st.rerun()
                    
                    # if hashed_password == correct_hash:
                    #     st.session_state.authenticated = True
                    #     st.success("✅ Login successful! Redirecting...")
                    #     time.sleep(1)
                    #     st.rerun()
                    else:
                        st.error("❌ Incorrect password. Please try again.")
                        
                except Exception as e:
                    logger.error(f"Authentication error: {e}")
                    st.error("❌ Authentication error. Please contact administrator.")
            
            elif submit_button and not password:
                st.warning("⚠️ Please enter a password.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.stop()  # Stop execution if not authenticated

class SimpleRAGSystem:
    def __init__(self):
        """Initialize the RAG system"""
        try:
            logger.info("Initializing RAG system...")
            # Initialize OpenAI
            openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                st.error("❌ OpenAI API key not found. Please check your secrets configuration.")
                st.stop()
            
            http = httpx.Client(trust_env=False)
            self.openai_client = OpenAI(api_key=openai_api_key, http_client=http)
            self.openai_client.models.list()
            
            logger.info(f"Connected to OpenAI Models")

            # Initialize Pinecone
            pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv('PINECONE_API_KEY')
            
            if not pinecone_api_key:
                st.error("❌ Pinecone credentials not found. Please check your secrets configuration.")
                st.stop()
                        
            # Connect to Pinecone index
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index_name = st.secrets.get("PINECONE_INDEX_NAME") or os.getenv('PINECONE_INDEX_NAME', 'rag-index')
            self.index = self.pc.Index(self.index_name)
                        
            # Test connection and View stats for the index
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone index {self.index_name} with {stats['total_vector_count']} documents")

        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            st.error(f"❌ Failed to initialize RAG system: {str(e)}")
            st.stop()
    
    def create_embedding(self, text):
        """Create OpenAI embedding"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            logging.info("Embeddings created...")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            st.error(f"Error creating embedding: {e}")
            return None
        
    def check_existing_documents(self, idx):
        """Check for overlap with documents already in the db"""
        resp = self.index.fetch(ids=idx)
        print(resp)
    
    def search_documents(self, query, top_k=5):
        """Search for relevant documents"""
        try:
            logger.info(f"Searching for: {query[:50]}...")
            query_embedding = self.create_embedding(query)
            if not query_embedding:
                return []
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            contexts = []
            for match in results['matches']:
                if match["score"] >= 0.60:
                    contexts.append({
                        'text': match['metadata']['embed_text'],
                        'source': match['metadata']['source'],
                        'score': round(match['score'], 3)
                    })
            
            logger.info(f"Found {len(contexts)} relevant documents")
            return contexts
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            st.error(f"Error searching documents: {e}")
            return []
    
    def generate_answer(self, query, contexts):
        """Generate answer using GPT"""
        try:
            if contexts:
                context_text = "\n\n".join([f"{ctx['text']}\n" for ctx in contexts])
                prompt = f"""Answer the question based on the context below. 
                    If the context doesn't contain the answer, say so.
                    
                    # Context:
                    # {context_text}

                    # Question: {query}
                    # Answer:"""
            else:
                prompt = f"I don't have relevant information to answer: {query}."

            completion = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.8,
                max_tokens=400,
                )  
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            st.error(f"Error generating answer: {e}")
            return "Sorry, I encountered an error generating a response."

    def create_vector_id(self, text):
        """Create a md5 id for the vector"""
        md5_value = hashlib.md5(text.encode("utf-8")).hexdigest()
        return md5_value

    def add_document(self, payload):
        """Add document to knowledge base"""
        try:
            if not payload["combined_text"].strip():
                return False
                        
            embedding = self.create_embedding(payload["combined_text"])
            if not embedding:
                return False
            
            doc_id = self.create_vector_id(text=payload["combined_text"])

            self.index.upsert(
                vectors=[
                    {
                    "id": doc_id, 
                    "values": embedding, 
                    "metadata": {
                        "source": payload["source"],
                        "embed_text": payload["combined_text"],
                        "source_file": payload["source_file"],
                        "upload_timestamp": datetime.now().isoformat()
                    }
                    },
                ]
                )
            return True
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            st.error(f"Error adding document: {e}")
            return False
    
    def wipe_database(self):
        """Delete all vectors from the index"""
        try:
            logger.warning("Attempting to wipe database...")
            # Delete all vectors by deleting everything in the index
            self.index.delete(delete_all=True)
            logger.warning("Database wiped successfully")
            return True
        except Exception as e:
            logger.error(f"Error wiping database: {e}")
            st.error(f"Error wiping database: {e}")
            return False
        
    def vector_exists(self, vector_id: str) -> bool:
        """Return True if a vector with id exists."""
        try:
            resp = self.index.fetch(ids=[vector_id])
            return bool(resp and resp.get("vectors") and vector_id in resp["vectors"])
        except Exception as e:
            logger.error(f"Error checking vector existence: {e}")
            return False
        
    def reindex_from_dataframe(self, df: pd.DataFrame, source_file_name: str) -> int:
        """
        Wipe the index and re-upload from the provided DataFrame.
        Expects columns 'q' and 'a'.
        Returns the number of successfully added FAQs.
        """
        if df is None or df.empty:
            st.error("❌ No data to re-index.")
            return 0

        # 1) wipe
        with st.spinner("🧹 Wiping existing index..."):
            if not self.wipe_database():
                return 0

        # 2) ingest
        success_count = 0
        progress_bar = st.progress(0)
        total = len(df)

        for idx, row in df.iterrows():
            q_text = str(row.get("q", "")).strip()
            a_text = str(row.get("a", "")).strip()
            if not q_text or not a_text:
                progress_bar.progress((idx + 1) / total)
                continue

            combined_text = f"Question: {q_text} Answer: {a_text}"
            doc_id = self.create_vector_id(combined_text)

            # optional: skip duplicates (shouldn’t occur after wipe, but safe)
            if self.vector_exists(doc_id):
                progress_bar.progress((idx + 1) / total)
                continue

            added = self.add_document(payload={
                "question": q_text,
                "answer": a_text,
                "combined_text": combined_text,
                "source": f"FAQ #{idx + 1}",
                "source_file": source_file_name
            })
            if added:
                success_count += 1

            progress_bar.progress((idx + 1) / total)

        return success_count

# Initialize RAG system
@st.cache_resource
def init_rag_system():
    return SimpleRAGSystem()

def main():
    simple_auth()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Chatbot Demo</h1>', unsafe_allow_html=True)
    st.markdown("### Ask questions and get answers from your knowledge base!")
    
    # Initialize RAG system
    with st.spinner("🔄 Initializing RAG system..."):
        rag_system = init_rag_system()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your RAG-powered FAQ assistant"}
        ]
    
    # Sidebar with all components
    with st.sidebar:
        st.markdown("---")
        st.success("🔓 Authenticated")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
        
        # Database Status section
        st.markdown("---")
        st.header("📈 Database Status")
        
        try:
            # Get database stats
            stats = rag_system.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            index_fullness = stats.get('index_fullness', 0)
            
            # Display stats in a nice format
            st.markdown(
                f"""
                <div class="stats-box">
                    <h4>📊 Knowledge Base Stats</h4>
                    <p><strong>Total FAQs Uploaded:</strong> {total_vectors:,}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"❌ Error fetching database stats: {str(e)}")
            logger.error(f"Database stats error: {e}")
        
        st.markdown("---")        
        # CSV Upload section
        st.header("📊 Upload more FAQ documents")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to add multiple FAQs at once"
        )
        
        if uploaded_file is not None:
            try:
                # Name of CSV file
                source_file_name = uploaded_file.name
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                # Display CSV info
                st.write(f"📋 **File:** {source_file_name}")
                st.write(f"📊 **Rows:** {len(df)}, **Columns:** {len(df.columns)}")
                
                # Upload button
                if st.button("🚀 Upload FAQs from CSV", use_container_width=True):
                    with st.spinner(f"Processing {len(df)} documents..."):
                        success_count = 0
                        progress_bar = st.progress(0)
                        
                        for idx, row in df.iterrows():
                            # Get question text content
                            q_text_content = str(row["q"]).strip()
                            
                            # Get answer text content
                            a_text_content = str(row["a"]).strip()

                            source_faq = f"FAQ #{idx + 1}"

                            combined_text_content = f"Question: {q_text_content} Answer: {a_text_content}"

                            # Skip rows already present
                            # rag_system.check_existing_documents(idx=text_id)

                            success = rag_system.add_document(payload={
                                "question": q_text_content,
                                "answer": a_text_content,
                                "combined_text": combined_text_content,
                                "source": source_faq,
                                "source_file": source_file_name
                            })
                            if success:
                                success_count += 1

                            # Update progress
                            progress_bar.progress((idx + 1) / len(df))
                        
                        # Show results
                        st.success(f"✅ Successfully uploaded {success_count}/{len(df)} FAQs!")
                        logger.info(f"CSV upload complete: {success_count}/{len(df)} FAQs added")
                        
                        # Show updated database stats
                        try:
                            updated_stats = rag_system.index.describe_index_stats()
                            new_total = updated_stats.get('total_vector_count', 0)
                            st.info(f"📊 Database now contains {new_total + 1:,} FAQs total")
                        except Exception as e:
                            logger.error(f"Error getting updated stats: {e}")
                        
                        # Refresh the page to update stats
                        time.sleep(2)
                        st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Error processing CSV file: {str(e)}")
                logger.error(f"CSV upload error: {e}")
                
        with st.sidebar:
            st.markdown("---")
            st.header("🗃️ Database Management")

            # Wipe database (2-click confirm)
            if st.button("🗑️ Wipe Database", use_container_width=True, help="Delete all FAQs from the knowledge base"):
                if st.session_state.get('confirm_wipe'):
                    db_stats = rag_system.index.describe_index_stats()
                    db_total = db_stats.get('total_vector_count', 0)
                    if db_total > 0:
                        with st.spinner("Wiping database..."):
                            if rag_system.wipe_database():
                                st.success("✅ Database wiped successfully!")
                                try:
                                    updated_stats = rag_system.index.describe_index_stats()
                                    remaining_docs = updated_stats.get('total_vector_count', 0)
                                    st.info(f"📊 Database now contains {remaining_docs:,} FAQs")
                                except Exception as e:
                                    logger.error(f"Error getting updated stats: {e}")

                                st.session_state.messages = [
                                    {"role": "assistant", "content": "Database has been wiped. Upload or re-index to get started!"}
                                ]
                                time.sleep(2)
                                st.rerun()
                    else:
                        st.error("❌ DB empty, nothing to wipe.")
                    st.session_state.confirm_wipe = False
                else:
                    st.session_state.confirm_wipe = True
                    st.warning("⚠️ Click again to confirm database wipe")

            # Re-index control
            st.markdown("### 🔄 Re-index from CSV")
            st.caption("Wipes the index and re-uploads from a CSV with columns **q** and **a**.")
            reindex_file = st.file_uploader(
                "Choose a CSV to Re-index",
                type=['csv'],
                key="reindex_csv_uploader",
                help="This file will be used to rebuild the knowledge base from scratch"
            )

            if st.button("🔄 Re-index Now", use_container_width=True):
                # Prefer the dedicated reindex_file; fall back to the general upload if available
                csv_to_use = None
                csv_name = None

                if reindex_file is not None:
                    csv_to_use = reindex_file
                    csv_name = reindex_file.name
                elif 'uploaded_file' in st.session_state and st.session_state['uploaded_file'] is not None:
                    csv_to_use = st.session_state['uploaded_file']
                    csv_name = csv_to_use.name

                if csv_to_use is None:
                    st.error("❌ Please upload a CSV in the 'Re-index from CSV' section first.")
                else:
                    try:
                        df_re = pd.read_csv(csv_to_use)
                        with st.spinner(f"Re-indexing from {csv_name}..."):
                            added = rag_system.reindex_from_dataframe(df_re, csv_name)
                            st.success(f"✅ Re-index complete. Uploaded {added}/{len(df_re)} FAQs.")
                            try:
                                stats_after = rag_system.index.describe_index_stats()
                                total_after = stats_after.get('total_vector_count', 0)
                                st.info(f"📊 Database now contains {total_after:,} FAQs")
                            except Exception as e:
                                logger.error(f"Error getting updated stats: {e}")
                            time.sleep(2)
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error reading CSV: {e}")

    
    # Main chat interface
    st.header("💬 Chat with your Knowledge Base")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about us..."):
        logger.info(f"User query: {prompt[:50]}...")
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching knowledge base..."):
                contexts = rag_system.search_documents(prompt)
                answer = rag_system.generate_answer(prompt, contexts)
                
                # Display answer
                st.write(answer)
                
                # Display context info
                if contexts:
                    st.caption(f"📚 Found {len(contexts)} relevant documents")
                    
                    # Show sources in expander
                    with st.expander("🔍 View Sources"):
                        for i, ctx in enumerate(contexts, 1):
                            st.write(f"**{i}. {ctx['source']}**, Similarity Score: {ctx['score']}")
                            st.write(ctx['text'][:300] + "..." if len(ctx['text']) > 300 else ctx['text'])
                            if i < len(contexts):
                                st.markdown("---")
                else:
                    st.caption("📭 No relevant documents found in knowledge base")
                
                # Create full response for session state
                context_info = f"📚 Found {len(contexts)} relevant documents" if contexts else "📭 No relevant documents found"
                full_answer = f"{answer}\n\n*{context_info}*"
                
                # Add to session state
                st.session_state.messages.append({"role": "assistant", "content": full_answer})

if __name__ == "__main__":
    main()