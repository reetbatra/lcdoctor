"""Small single-purpose chains the agent leans on."""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

REWRITE = PromptTemplate.from_template(
    "Rewrite this support question so it matches how the docs phrase things:\n\n{question}"
)


def rewrite_question(question: str) -> str:
    chain = LLMChain(llm=ChatOpenAI(model="gpt-4o-mini"), prompt=REWRITE)
    return chain.run(question=question)
