"""The newer LangGraph version of the same agent, half migrated."""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from retriever import search_docs as _search


@tool
def search_docs(question: str) -> str:
    """Search the product docs."""
    return _search(question)


graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[search_docs])
