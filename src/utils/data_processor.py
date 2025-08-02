import os
import pypdf
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdfs(pdf_directory: str) -> str:
    """
    Extracts text from all PDF files in a given directory.

    Args:
        pdf_directory (str): The path to the directory containing PDF files.

    Returns:
        str: A single string containing all the extracted text from all PDFs.
    """
    all_text = ""
    if not os.path.exists(pdf_directory):
        logging.warning(f"Directory not found: {pdf_directory}")
        return all_text

    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    logging.info(f"Found {len(pdf_files)} PDF(s) in '{pdf_directory}'")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                logging.info(f"Reading text from {pdf_file}...")
                for page_num, page in enumerate(reader.pages):
                    all_text += page.extract_text() or ""
                all_text += "\n\n--- End of Document: " + pdf_file + " ---\n\n"
        except Exception as e:
            logging.error(f"Could not read {pdf_file}: {e}")
    
    return all_text

def clean_extracted_text(text: str) -> str:
    """
    Performs basic cleaning on the extracted text.

    Args:
        text (str): The raw text extracted from PDFs.

    Returns:
        str: The cleaned text.
    """
    logging.info("Cleaning extracted text...")
    # Remove multiple consecutive newlines and replace with a single one
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Remove extra whitespace from the beginning and end of each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    logging.info("Text cleaning complete.")
    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Splits a large text into smaller chunks.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The maximum size of each chunk (in characters).
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list: A list of text chunks (strings).
    """
    logging.info(f"Chunking text with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    logging.info(f"Successfully created {len(chunks)} chunks.")
    return chunks

if __name__ == '__main__':
    # This block allows you to run this script directly for testing
    # Define the path to your raw PDFs
    # The path is constructed relative to the project's root directory
    
    # Assuming the script is run from the root of the project (textbook_qa_bot/)
    # If you run it from within src/utils/, you might need to adjust the path
    # e.g., pdf_dir = "../../data/raw_pdfs"
    pdf_dir = "data/raw_pdfs"
    
    # 1. Extract text
    raw_text = extract_text_from_pdfs(pdf_dir)
    
    if raw_text:
        # 2. Clean text
        cleaned_text = clean_extracted_text(raw_text)
        
        # 3. Chunk text
        text_chunks = chunk_text(cleaned_text)
        
        # You can print a sample to see the result
        print(f"\nTotal chunks created: {len(text_chunks)}")
        if text_chunks:
            print("\n--- Sample Chunk ---")
            print(text_chunks[0])
            print("--------------------")
    else:
        print("No text was extracted. Please check the 'data/raw_pdfs' directory and ensure it contains PDF files.")

