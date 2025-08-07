# embedder.py
# Generate and store embeddings in ChromaDB from text chunks

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
from typing import List

def generate_embeddings(chunks: List[str], persist_directory: str = "db") -> Chroma:
    """
    Generates embeddings for the text chunks and stores them in ChromaDB.
    :param chunks: List of text chunks
    :param persist_directory: Directory path to persist ChromaDB
    :return: Chroma vector store instance
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

if __name__ == "__main__":
    sample_chunks = [
        "The company's revenue in Q1 was $5 million.",
        "Expenses dropped by 10% year-over-year.",
        "The net profit margin improved to 20%."
    ]
    db = generate_embeddings(sample_chunks)
    print("Embeddings stored in ChromaDB.")
