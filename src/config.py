import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") # <-- Add this line

LLM_MODEL_NAME = "gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"g