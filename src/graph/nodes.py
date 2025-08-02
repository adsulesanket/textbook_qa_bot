import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

# Import from our project structure
from src.vector_store import get_vector_store, build_or_get_collection
from src.config import GOOGLE_API_KEY, TAVILY_API_KEY, LLM_MODEL_NAME
from .state import GraphState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize LLM and Tools
llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model=LLM_MODEL_NAME, temperature=0)
web_search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY, max_results=3)

# --- Node Functions ---

def retrieve(state: GraphState) -> GraphState:
    """Retrieves documents from the vector store."""
    logging.info("Node: Retrieving documents...")
    question = state["question"]
    client, embedding_func = get_vector_store()
    collection = build_or_get_collection(client, embedding_func)
    results = collection.query(query_texts=[question], n_results=3)
    return {"documents": results['documents'][0], "question": question}

def grade_documents(state: GraphState) -> GraphState:
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    logging.info("Node: Grading documents...")
    question = state["question"]
    documents = state["documents"]

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keywords related to the user question, grade it as relevant.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

        Here is the retrieved document:
        \n---\n
        {document}
        \n---\n
        Here is the user question: {question}
        """,
        input_variables=["document", "question"],
    )

    parser = JsonOutputParser()
    chain = prompt | llm | parser

    # Grade each document
    filtered_docs = []
    for d in documents:
        score = chain.invoke({"document": d, "question": question})
        grade = score['score']
        if grade.lower() == "yes":
            logging.info("Document is relevant.")
            filtered_docs.append(d)
        else:
            logging.info("Document is NOT relevant.")
            continue
    return {"documents": filtered_docs, "question": question}

def generate(state: GraphState) -> GraphState:
    """Generates an answer using the retrieved documents."""
    logging.info("Node: Generating answer from documents...")
    question = state["question"]
    documents = state["documents"]
    
    prompt_template = """You are an expert AI tutor. Use the following retrieved context to answer the user's question.
    If you don't know the answer from the context, state that you don't know. Respond in the same language as the question.

    Context:
    {context}
    
    Question:
    {question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | llm
    
    context_str = "\n\n".join(documents)
    generation = chain.invoke({"context": context_str, "question": question}).content
    return {"generation": generation}

def web_search(state: GraphState) -> GraphState:
    """Performs a web search for the user's question."""
    logging.info("Node: Performing web search...")
    question = state["question"]
    search_results = web_search_tool.invoke({"query": question})
    return {"web_search_results": search_results, "question": question}
    
def generate_from_web(state: GraphState) -> GraphState:
    """Generates an answer using the web search results."""
    logging.info("Node: Generating answer from web search...")
    question = state["question"]
    web_results = state["web_search_results"]

    prompt_template = """You are an expert AI assistant. Use the following web search results to answer the user's question.
    Provide a concise answer based on the information found. Respond in the same language as the question.

    Web Search Results:
    {web_results}
    
    Question:
    {question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["web_results", "question"])
    chain = prompt | llm
    
    generation = chain.invoke({"web_results": web_results, "question": question}).content
    return {"generation": generation}
