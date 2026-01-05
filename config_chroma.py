"""
Configuration file for ChromaDB implementation.
This ensures consistent settings across all ChromaDB modules.
"""

# ChromaDB Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "master_bmr_compliance"

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Chunking Configuration
CHUNK_SIZE = 300  # Number of words per chunk
CHUNK_OVERLAP = 50  # Number of overlapping words

# Batch Processing
BATCH_SIZE = 32  # Embedding generation batch size
INSERT_BATCH_SIZE = 500  # ChromaDB insertion batch size

# Input File
INPUT_FILE_PATH = "Master Compliance BMR.pdf"
