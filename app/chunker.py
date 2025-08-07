# chunker.py
# Split cleaned text into manageable, overlapping chunks for embedding

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Splits the input text into chunks using RecursiveCharacterTextSplitter.
    :param text: Cleaned text
    :param chunk_size: Maximum characters per chunk
    :param chunk_overlap: Characters to overlap between chunks for context
    :return: List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    sample_text = "Revenue increased by 15% in Q1 2023.\nNet income rose to $2 million.\nFuture outlook is positive."
    chunks = chunk_text(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")
