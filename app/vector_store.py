# vector_store.py
# Module to manage storage and retrieval from ChromaDB

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


def load_vector_store(persist_directory: str = "db") -> Chroma:
    """
    Loads the Chroma vector store from the persisted directory.
    :param persist_directory: Directory path where ChromaDB is stored
    :return: Loaded Chroma vector store instance
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    return vectordb


if __name__ == "__main__":
    db = load_vector_store()
    print("Chroma vector store loaded successfully.")
