import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings  # Free embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import tempfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Store documents in session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings_cache' not in st.session_state:
    st.session_state.embeddings_cache = []

@st.cache_resource
def get_embeddings():
    """Get free HuggingFace embeddings (no API key needed)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def get_vector_store():
    """Return a simple document store"""
    return st.session_state.documents

def build_or_get_collection():
    """Get the document collection"""
    return get_vector_store()

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
    """Search documents using simple similarity"""
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