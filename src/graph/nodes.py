import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
import streamlit as st
from src.config import GOOGLE_API_KEY, LLM_MODEL_NAME, TEMPERATURE
from src.vector_store import search_documents

# Initialize Google Gemini model
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=TEMPERATURE,
        google_api_key=GOOGLE_API_KEY
    )

def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve documents from vector store"""
    query = state.get("question", "")
    
    # Search documents using our vector store
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
            # Include document if grading fails
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

def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """Perform web search (placeholder - requires Tavily API)"""
    # For now, just return empty results
    # You can implement actual web search later with Tavily API
    state["web_results"] = []
    return state

def generate_from_web(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate answer from web search results"""
    # Placeholder implementation
    question = state.get("question", "")
    state["answer"] = f"Web search is not implemented yet for: {question}"
    return state