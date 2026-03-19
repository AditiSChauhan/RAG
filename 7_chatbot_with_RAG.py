# ==============================
# STREAMLIT CHAT UI WITH LANGGRAPH + GROQ
# ==============================

import streamlit as st
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentState

from typing import Literal
from pydantic import BaseModel

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# -----------------------------
# MODEL SETUP
# -----------------------------
model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

model_with_tools = model.bind_tools([DuckDuckGoSearchResults()])

# -----------------------------
# STATE
# -----------------------------
class State(AgentState):
    iteration: int

class Decision(BaseModel):
    decision: Literal["yes", "no"]

model_decision = model.with_structured_output(Decision)

# -----------------------------
# NODES
# -----------------------------
def ask_llm(state: State):
    messages = state.get("messages", [])

    if not messages:
        raise ValueError("No messages found")

    # LLM processes full conversation
    response = model_with_tools.invoke(messages)

    return {
        "messages": messages + [response]
    }


def show_answer(state: State):
    return {
        "iteration": state.get("iteration", 0) + 1
    }


def end_condition(state: State) -> Literal["yes", "no"]:
    decision = model_decision.invoke(
        state["messages"] + [SystemMessage(content="Should we end conversation?")]
    )
    return decision.decision

# -----------------------------
# GRAPH BUILD
# -----------------------------
graph = StateGraph(State)

graph.add_node("ask_llm", ask_llm)
graph.add_node("tools", ToolNode(tools=[DuckDuckGoSearchResults()]))
graph.add_node("show_answer", show_answer)

graph.add_edge(START, "ask_llm")

graph.add_conditional_edges(
    "ask_llm",
    tools_condition,
    {
        "tools": "tools",
        END: "show_answer",
    },
)

graph.add_edge("tools", "show_answer")
graph.add_edge("show_answer", END)

workflow = graph.compile()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬")
st.title("💬 LangGraph Chatbot (Groq)")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Chat input
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message immediately
    st.chat_message("user").write(user_input)

    # Add user message to state
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Run LangGraph
    try:
        result = workflow.invoke({
            "messages": st.session_state.messages,
            "iteration": 0
        })

        # Update memory
        st.session_state.messages = result["messages"]

        # Show assistant response
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