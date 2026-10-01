"""Docs support agent: answers questions about a product's docs, with tools."""

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from retriever import search_docs


def build_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [
        Tool(
            name="search_docs",
            func=search_docs,
            description="Search the product docs. Input is a plain-English question.",
        )
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
    )
