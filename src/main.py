import logging
from fastapi import FastAPI
from pydantic import BaseModel
from src.graph.builder import build_graph
from src.graph.state import GraphState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the FastAPI app
app = FastAPI(
    title="Textbook Q&A Agent",
    description="An API for asking questions to an agent with textbook knowledge and web search fallback.",
    version="1.0.0"
)

# Build the compiled LangGraph app once when the server starts
langgraph_app = build_graph()

# Define the request body model
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    """
    Receives a question, runs it through the LangGraph agent, and returns the answer.
    """
    try:
        # Define the initial input for the graph
        inputs = GraphState(
            question=query.question, 
            documents=[], 
            web_search_results="", 
            generation=""
        )
        
        # Invoke the graph to get the final result
        final_state = langgraph_app.invoke(inputs)
        
        # Return the final generated answer
        return {"answer": final_state['generation']}

    except Exception as e:
        logging.error(f"An error occurred in the /ask endpoint: {e}", exc_info=True)
        return {"error": "An internal error occurred."}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Textbook Q&A Agent API. Please use the /ask endpoint to post your questions."}
