import streamlit as st
from src.graph.builder import build_graph
from src.graph.state import GraphState

# --- App Configuration ---
st.set_page_config(page_title="Textbook Q&A Chatbot", page_icon="📚")
st.title("📚 Textbook Q&A Chatbot")
st.write("Ask a question about your textbooks. The agent can also search the web if needed.")

# --- Initialize the LangGraph Agent ---
# Use a cached function to build the graph only once
@st.cache_resource
def load_agent():
    return build_graph()

app = load_agent()

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Ask your question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Define the initial state for the graph
            inputs = GraphState(question=prompt, documents=[], web_search_results="", generation="")
            # Invoke the agent
            final_state = app.invoke(inputs)
            response = final_state['generation']
            st.markdown(response)
    
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})