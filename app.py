import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your modules
try:
    from src.vector_store import add_documents_to_store, search_documents
    from src.nodes import retrieve, grade_documents, generate
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

def main():
    st.title("📚 Textbook Q&A Bot")
    st.write("Upload your textbooks and ask questions using Google Gemini!")
    
    # Check for Google API key
    if not os.getenv("GOOGLE_API_KEY"):
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
            # Add documents to vector store
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
                
                # Retrieve documents
                state = retrieve(state)
                
                # Grade documents
                state = grade_documents(state)
                
                # Generate answer
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
                st.error("Make sure your Google API key is valid and has Gemini API access enabled.")

if __name__ == "__main__":
    main()