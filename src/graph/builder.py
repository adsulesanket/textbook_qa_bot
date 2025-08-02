from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import retrieve, grade_documents, generate, web_search, generate_from_web

def decide_to_generate(state: GraphState) -> str:
    """
    Determines whether to generate an answer from documents or perform a web search.
    """
    if not state["documents"]:
        # All documents were filtered out as not relevant
        print("---DECISION: No relevant documents found. Routing to web search.---")
        return "web_search"
    else:
        # We have relevant documents, so generate an answer
        print("---DECISION: Relevant documents found. Routing to generation.---")
        return "generate"

def build_graph():
    """
    Builds and compiles the new agentic LangGraph workflow.
    """
    workflow = StateGraph(GraphState)

    # Add the nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_from_web", generate_from_web)

    # Build the graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate_from_web")
    workflow.add_edge("generate", END)
    workflow.add_edge("generate_from_web", END)

    # Compile the graph
    app = workflow.compile()
    return app
