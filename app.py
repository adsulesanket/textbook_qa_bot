import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import tempfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL_NAME = "gemini-1.5-flash"
TEMPERATURE = 0.1

# Store documents in session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings_cache' not in st.session_state:
    st.session_state.embeddings_cache = []

@st.cache_resource
def get_embeddings():
    """Get free HuggingFace embeddings"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def get_llm():
    """Initialize Google Gemini model"""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=TEMPERATURE,
        google_api_key=GOOGLE_API_KEY
    )

def add_documents_to_store(uploaded_files):
    """Add uploaded PDF documents to the store"""
    if not uploaded_files:
        return None
        
    # Clear existing documents
    st.session_state.documents = []
    st.session_state.embeddings_cache = []
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Add to session state
            st.session_state.documents.extend(split_docs)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            
        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)
    
    return st.session_state.documents

def search_documents(query, k=5):
    """Search documents using similarity"""
    if not st.session_state.documents:
        return []
    
    try:
        embeddings_model = get_embeddings()
        
        # Get query embedding
        query_embedding = embeddings_model.embed_query(query)
        
        # Calculate embeddings for documents if not cached
        if len(st.session_state.embeddings_cache) != len(st.session_state.documents):
            st.session_state.embeddings_cache = []
            with st.spinner("Computing document embeddings..."):
                for doc in st.session_state.documents:
                    doc_embedding = embeddings_model.embed_documents([doc.page_content])[0]
                    st.session_state.embeddings_cache.append(doc_embedding)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(st.session_state.embeddings_cache):
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities.append((similarity, i))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        top_docs = []
        for similarity, idx in similarities[:k]:
            if similarity > 0.1:  # Minimum similarity threshold
                top_docs.append(st.session_state.documents[idx])
        
        return top_docs
        
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return []

def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve documents from vector store"""
    query = state.get("question", "")
    documents = search_documents(query, k=5)
    
    state["documents"] = documents
    state["context"] = "\n\n".join([doc.page_content for doc in documents])
    
    return state

def grade_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """Grade retrieved documents for relevance"""
    question = state.get("question", "")
    documents = state.get("documents", [])
    
    if not documents:
        state["relevant_documents"] = []
        return state
    
    llm = get_llm()
    
    # Simple grading prompt
    grade_prompt = PromptTemplate(
        template="""You are a grader assessing relevance of retrieved documents to a user question.
        
        Question: {question}
        
        Document: {document}
        
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
        Respond with just 'yes' or 'no'.""",
        input_variables=["question", "document"]
    )
    
    relevant_docs = []
    for doc in documents:
        try:
            prompt_text = grade_prompt.format(question=question, document=doc.page_content)
            grade = llm.invoke(prompt_text)
            if "yes" in grade.content.lower():
                relevant_docs.append(doc)
        except Exception as e:
            st.error(f"Error grading document: {e}")
            relevant_docs.append(doc)
    
    state["relevant_documents"] = relevant_docs
    return state

def generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate answer from relevant documents"""
    question = state.get("question", "")
    relevant_docs = state.get("relevant_documents", [])
    
    if not relevant_docs:
        state["answer"] = "I couldn't find relevant information to answer your question."
        return state
    
    llm = get_llm()
    
    # Create context from relevant documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Generate answer prompt
    answer_prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        
        Question: {question}
        
        Context: {context}
        
        Answer:""",
        input_variables=["question", "context"]
    )
    
    try:
        prompt_text = answer_prompt.format(question=question, context=context)
        response = llm.invoke(prompt_text)
        state["answer"] = response.content
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        state["answer"] = "Sorry, I encountered an error while generating the answer."
    
    return state

def main():
    st.title("📚 Textbook Q&A Bot")
    st.write("Upload your textbooks and ask questions using Google Gemini!")
    
    # Check for Google API key
    if not GOOGLE_API_KEY:
        st.error("Please set your GOOGLE_API_KEY in the secrets or environment variables.")
        st.info("Get your free Google API key from: https://makersuite.google.com/app/apikey")
        st.stop()
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            docs = add_documents_to_store(uploaded_files)
            if docs:
                st.success(f"Processed {len(uploaded_files)} files with {len(docs)} chunks successfully!")
            else:
                st.error("Failed to process documents.")
    
    # Question input
    question = st.text_input("Ask a question about your textbooks:")
    
    if st.button("Get Answer") and question:
        if not uploaded_files:
            st.warning("Please upload some PDF files first.")
            return
            
        with st.spinner("Searching for answer..."):
            try:
                # Create state
                state = {"question": question}
                
                # Process question
                state = retrieve(state)
                state = grade_documents(state)
                state = generate(state)
                
                # Display answer
                if state.get("answer"):
                    st.write("### Answer:")
                    st.write(state["answer"])
                    
                    # Show relevant documents
                    if state.get("relevant_documents"):
                        with st.expander("View source documents"):
                            for i, doc in enumerate(state["relevant_documents"]):
                                st.write(f"**Document {i+1}:**")
                                st.write(doc.page_content[:500] + "...")
                                st.write("---")
                else:
                    st.error("Could not generate an answer.")
                    
            except Exception as e:
                st.error(f"Error processing question: {e}")

if __name__ == "__main__":
    main()
    