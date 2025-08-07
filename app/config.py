# config.py

# Path to store or retrieve ChromaDB vectors
db_directory = "db/chromadb"

# Chunking configuration
chunk_size = 500
chunk_overlap = 50

# Embedding model to use (change if using OpenAI or other embeddings)
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Mistral model settings (if using HuggingFaceHub)
mistral_repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
mistral_temperature = 0.5
mistral_max_tokens = 500
