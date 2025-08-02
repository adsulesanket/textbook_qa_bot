from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question (str): The user's question.
        documents (List[str]): The list of retrieved documents from the vector store.
        web_search_results (str): The results from the web search.
        generation (str): The LLM's generated answer.
    """
    question: str
    documents: List[str]
    web_search_results: str
    generation: str
