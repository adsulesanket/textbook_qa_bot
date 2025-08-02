import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # Optional for web search

# Model Configuration
LLM_MODEL_NAME = "gemini-1.5-flash"  # Google Gemini model

# Vector Store Configuration
VECTOR_STORE_TYPE = "faiss"  # Using FAISS instead of ChromaDB

# Other Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 2000
TEMPERATURE = 0.1