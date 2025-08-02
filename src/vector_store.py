import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import tempfile

@st.cache_resource
def get_embeddings():
    """Get OpenAI embeddings"""
    return OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

@st.cache_resource
def get_vector_store():
    """Initialize empty FAISS vector store"""
    embeddings = get_embeddings()
    
    # Create a dummy document to initialize FAISS
    from langchain.schema import Document
    dummy_doc = Document(page_content="Initialization document", metadata={})
    
    # Create FAISS vector store with dummy document
    vector_store = FAISS.from_documents([dummy_doc], embeddings)
    
    return vector_store

def build_or_get_collection():
    """Get the FAISS collection"""
    return get_vector_store()

def add_documents_to_store(uploaded_files):
    """Add uploaded PDF documents to the vector store"""
    if not uploaded_files:
        return None
        
    embeddings = get_embeddings()
    all_documents = []
    
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
            all_documents.extend(split_docs)
            
        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)
    
    if all_documents:
        # Create new FAISS vector store with documents
        vector_store = FAISS.from_documents(all_documents, embeddings)
        return vector_store
    
    return None

def search_documents(vector_store, query, k=5):
    """Search documents in the vector store"""
    if vector_store is None:
        return []
    
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return []