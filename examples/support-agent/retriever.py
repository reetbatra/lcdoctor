"""Chunk the docs, embed them, and search them."""

from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.documents import Document

_store = None


def load_docs(path: str = "docs/") -> list[Document]:
    docs = DirectoryLoader(path, glob="**/*.md").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)


def search_docs(question: str) -> str:
    global _store
    if _store is None:
        _store = FAISS.from_documents(load_docs(), OpenAIEmbeddings())
    hits = _store.similarity_search(question, k=4)
    return "\n\n".join(h.page_content for h in hits)
