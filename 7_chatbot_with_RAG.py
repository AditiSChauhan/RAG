# ==============================
# STREAMLIT CHAT UI WITH LANGGRAPH + GROQ + RAG
# ==============================

import streamlit as st
import os
from dotenv import load_dotenv
import tempfile

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentState

# ✅ RAG IMPORTS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from typing import Literal
from pydantic import BaseModel

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

# -----------------------------
# MODEL SETUP
# -----------------------------
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

model_with_tools = model.bind_tools([DuckDuckGoSearchResults()])

# -----------------------------
# RAG FUNCTION
# -----------------------------
def create_vector_db(uploaded_files):
    docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    vector_db = FAISS.from_documents(docs, embeddings)

    return vector_db

# -----------------------------
# STATE
# -----------------------------
class State(AgentState):
    iteration: int

class Decision(BaseModel):
    decision: Literal["yes", "no"]

model_decision = model.with_structured_output(Decision)

# -----------------------------
# NODES (UPDATED WITH RAG)
# -----------------------------
def ask_llm(state: State):
    messages = state.get("messages", [])
    vector_db = state.get("vector_db", None)

    if not messages:
        raise ValueError("No messages found")

    # 🔍 RAG: get context from documents
    if vector_db:
        query = messages[-1].content
        docs = vector_db.similarity_search(query, k=3)

        context = "\n\n".join([doc.page_content for doc in docs])

        # Inject context into LLM
        messages = messages + [
            HumanMessage(content=f"Use this context to answer:\n{context}")
        ]

    # LLM response
    response = model_with_tools.invoke(messages)

    return {
        "messages": messages + [response]
    }

# -----------------------------
# GRAPH BUILD
# -----------------------------
graph = StateGraph(State)

graph.add_node("ask_llm", ask_llm)
graph.add_node("tools", ToolNode(tools=[DuckDuckGoSearchResults()]))

graph.add_edge(START, "ask_llm")

graph.add_conditional_edges(
    "ask_llm",
    tools_condition,
    {
        "tools": "tools",
        END: END,
    },
)

graph.add_edge("tools", END)

workflow = graph.compile()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬")
st.title("💬 LangGraph Chatbot (Groq + RAG)")

# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# -----------------------------
# 📂 FILE UPLOAD (RAG UI)
# -----------------------------
st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Process Documents"):
    if uploaded_files:
        st.session_state.vector_db = create_vector_db(uploaded_files)
        st.sidebar.success("✅ Documents processed!")
    else:
        st.sidebar.warning("Upload at least one PDF")

# -----------------------------
# CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# -----------------------------
# CHAT INPUT
# -----------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message
    st.chat_message("user").write(user_input)

    # Add to memory
    st.session_state.messages.append(HumanMessage(content=user_input))

    try:
        result = workflow.invoke({
            "messages": st.session_state.messages,
            "vector_db": st.session_state.vector_db,
            "iteration": 0
        })

        # Update memory
        st.session_state.messages = result["messages"]

        # Show response
        last_msg = result["messages"][-1]
        st.chat_message("assistant").write(last_msg.content)

    except Exception as e:
        st.error(f"Error: {str(e)}")

# -----------------------------
# RESET BUTTON
# -----------------------------
if st.button("🔄 New Chat"):
    st.session_state.messages = []
    st.rerun()