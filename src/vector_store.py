import chromadb
from chromadb.utils import embedding_functions
import logging
import os

# Import our custom data processing functions and configuration
from src.utils.data_processor import extract_text_from_pdfs, clean_extracted_text, chunk_text
from src.config import EMBEDDING_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Define the path for the persistent ChromaDB database
CHROMA_DB_PATH = "data/chroma_db"
# Define the name of the collection to be used
COLLECTION_NAME = "textbook_collection"
# Define the path to the raw PDF data
PDF_DATA_PATH = "data/raw_pdfs"

def get_vector_store():
    """
    Initializes and returns a persistent ChromaDB client and a sentence transformer
    embedding function.

    Returns:
        tuple: A tuple containing the ChromaDB client and the embedding function.
    """
    logging.info("Initializing vector store...")
    
    # Initialize a persistent ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Initialize the embedding function using the model specified in config
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    
    logging.info(f"Vector store initialized with model: {EMBEDDING_MODEL_NAME}")
    return client, embedding_func

def build_or_get_collection(client, embedding_func):
    """
    Gets an existing vector collection or creates and populates a new one if it doesn't exist.

    Args:
        client: The ChromaDB client instance.
        embedding_func: The sentence transformer embedding function.

    Returns:
        chromadb.Collection: The ChromaDB collection object.
    """
    try:
        # First, try to get the collection. This will raise a ValueError if it doesn't exist.
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )
        logging.info(f"Successfully loaded existing collection '{COLLECTION_NAME}' with {collection.count()} items.")
        return collection
    except ValueError:
        # If the collection does not exist, this block will be executed.
        logging.info(f"Collection '{COLLECTION_NAME}' not found. Creating a new one.")
        
        # 1. Create the new collection
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"}
        )

        # 2. Process the data from PDFs
        logging.info("Starting data processing pipeline to populate new collection...")
        raw_text = extract_text_from_pdfs(PDF_DATA_PATH)
        if not raw_text:
            logging.error("No text extracted from PDFs. Cannot build collection.")
            # Clean up by deleting the empty collection that was just created
            client.delete_collection(name=COLLECTION_NAME)
            return None
        
        cleaned_text = clean_extracted_text(raw_text)
        text_chunks = chunk_text(cleaned_text)
        
        # Create unique IDs for each chunk
        chunk_ids = [str(i) for i in range(len(text_chunks))]
        
        # 3. Add the documents to the newly created collection in batches
        logging.info(f"Adding {len(text_chunks)} chunks to the collection. This may take a while...")
        batch_size = 100 
        for i in range(0, len(text_chunks), batch_size):
            collection.add(
                ids=chunk_ids[i:i+batch_size],
                documents=text_chunks[i:i+batch_size]
            )
            logging.info(f"Added batch {i//batch_size + 1}/{(len(text_chunks)//batch_size)+1}")

        logging.info(f"Successfully created and populated collection '{COLLECTION_NAME}'.")
        return collection

# --- Main Execution ---
if __name__ == '__main__':
    # This block allows you to run this script directly to build the database
    logging.info("Running vector store setup script...")
    
    # 1. Initialize the client and embedding function
    chroma_client, embedding_function = get_vector_store()
    
    # 2. Build the collection if it doesn't exist
    textbook_collection = build_or_get_collection(chroma_client, embedding_function)
    
    if textbook_collection:
        # 3. Verify the collection
        count = textbook_collection.count()
        logging.info(f"Verification complete. The collection '{COLLECTION_NAME}' contains {count} items.")
        
        # Example query
        print("\n--- Performing a test query ---")
        results = textbook_collection.query(
            query_texts=["What were the major causes of the Revolt of 1857?"],
            n_results=2
        )
        print("Test query results:")
        print(results['documents'])
        print("--------------------------")
    else:
        logging.error("Failed to build or load the vector store collection.")
