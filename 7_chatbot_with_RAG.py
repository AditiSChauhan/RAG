# ==============================
# STREAMLIT + LANGGRAPH + GROQ + RAG + AUTH + LOGOUT
# ==============================

import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentState

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# ENV
# -----------------------------
load_dotenv()

# -----------------------------
# DATABASE
# -----------------------------
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    username TEXT,
    role TEXT,
    message TEXT
)
""")

conn.commit()

# -----------------------------
# MODEL
# -----------------------------
model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

model_with_tools = model.bind_tools([DuckDuckGoSearchResults()])

# -----------------------------
# RAG FUNCTION
# -----------------------------
def create_vector_db(files):
    docs = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(docs, embeddings)

# -----------------------------
# STATE
# -----------------------------
class State(AgentState):
    pass

# -----------------------------
# NODE
# -----------------------------
def ask_llm(state: State):
    messages = state.get("messages", [])
    vector_db = state.get("vector_db")

    if vector_db:
        query = messages[-1].content
        docs = vector_db.similarity_search(query, k=3)

        context = "\n\n".join([d.page_content for d in docs])

        messages = messages + [
            HumanMessage(content=f"Use this context to answer:\n{context}")
        ]

    response = model_with_tools.invoke(messages)

    return {"messages": messages + [response]}

# -----------------------------
# GRAPH
# -----------------------------
graph = StateGraph(State)

graph.add_node("ask_llm", ask_llm)
graph.add_node("tools", ToolNode(tools=[DuckDuckGoSearchResults()]))

graph.add_edge(START, "ask_llm")

graph.add_conditional_edges(
    "ask_llm",
    tools_condition,
    {"tools": "tools", END: END}
)

graph.add_edge("tools", END)

workflow = graph.compile()

# =============================
# 🔐 AUTH SIDEBAR
# =============================
st.sidebar.title("🔐 Authentication")

auth_mode = st.sidebar.radio("Select", ["Login", "Register"])

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

# Register
if auth_mode == "Register":
    if st.sidebar.button("Register"):
        try:
            cursor.execute("INSERT INTO users VALUES (?, ?)", (username, password))
            conn.commit()
            st.sidebar.success("Registered successfully!")
        except:
            st.sidebar.error("User already exists")

# Login
if auth_mode == "Login":
    if st.sidebar.button("Login"):
        user = cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        ).fetchone()

        if user:
            st.session_state.user = username
            st.session_state.messages = []  # reset session
            st.sidebar.success("Login successful")
        else:
            st.sidebar.error("Invalid credentials")

# -----------------------------
# LOGOUT
# -----------------------------
if "user" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.write(f"👤 Logged in as: {st.session_state.user}")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.pop("user")
        st.session_state.pop("messages", None)
        st.session_state.pop("vector_db", None)
        st.sidebar.success("Logged out successfully")
        st.rerun()

# =============================
# MAIN APP
# =============================
st.set_page_config(page_title="Chatbot", page_icon="💬")
st.title("💬 Chatbot (RAG + Auth + Groq)")

# Block if not logged in
if "user" not in st.session_state:
    st.warning("Please login to continue")
    st.stop()

# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    rows = cursor.execute(
        "SELECT role, message FROM chats WHERE username=?",
        (st.session_state.user,)
    ).fetchall()

    msgs = []
    for role, msg in rows:
        if role == "user":
            msgs.append(HumanMessage(content=msg))
        else:
            msgs.append(AIMessage(content=msg))

    st.session_state.messages = msgs

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# -----------------------------
# FILE UPLOAD (RAG)
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.header("📂 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Process Documents"):
    if uploaded_files:
        st.session_state.vector_db = create_vector_db(uploaded_files)
        st.sidebar.success("Documents processed!")
    else:
        st.sidebar.warning("Upload at least one file")

# -----------------------------
# DISPLAY CHAT
# -----------------------------
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# -----------------------------
# CHAT INPUT
# -----------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    st.chat_message("user").write(user_input)

    st.session_state.messages.append(HumanMessage(content=user_input))

    # Save user message
    cursor.execute(
        "INSERT INTO chats VALUES (?, ?, ?)",
        (st.session_state.user, "user", user_input)
    )
    conn.commit()

    result = workflow.invoke({
        "messages": st.session_state.messages,
        "vector_db": st.session_state.vector_db
    })

    st.session_state.messages = result["messages"]

    last_msg = result["messages"][-1]

    st.chat_message("assistant").write(last_msg.content)

    # Save bot response
    cursor.execute(
        "INSERT INTO chats VALUES (?, ?, ?)",
        (st.session_state.user, "assistant", last_msg.content)
    )
    conn.commit()

# -----------------------------
# RESET CHAT
# -----------------------------
if st.button("🔄 New Chat"):
    cursor.execute(
        "DELETE FROM chats WHERE username=?",
        (st.session_state.user,)
    )
    conn.commit()

    st.session_state.messages = []
    st.rerun()