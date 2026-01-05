from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize embedding model (same as knowledge_base.py)
embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
print(f"Loaded embedding model: Qwen/Qwen3-Embedding-0.6B")
print(f"Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")

def retrieve_from_knowledge_base(query, index_file="Master Compliance BMR_faiss.index", 
                                 metadata_file="Master Compliance BMR_metadata.pkl", k=5):
    """
    Retrieve relevant chunks from knowledge base for a given query.
    
    Args:
        query (str): Parameter name or question to search for
        index_file (str): Path to FAISS index file
        metadata_file (str): Path to metadata pickle file
        k (int): Number of results to return
        
    Returns:
        list: Retrieved chunks with similarity scores
    """
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode(query, convert_to_numpy=True)
        
        # Load FAISS index
        index = faiss.read_index(index_file)
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        
        # Load metadata
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        logger.info(f"Loaded metadata for {len(metadata)} chunks")
        
        # Search FAISS index - retrieve more candidates
        query_vector = query_embedding.reshape(1, -1)
        distances, indices = index.search(query_vector, k * 2)
        
        # Get chunks with filtering
        chunks = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1 and distance < 2.0:  # Filter by L2 distance threshold
                chunk = metadata[idx].copy()
                chunk['similarity_score'] = float(distance)
                chunks.append(chunk)
        
        # Log results
        logger.info(f"Found {len(chunks)} relevant chunks for query: {query[:100]}...")
        for i, chunk in enumerate(chunks[:k], 1):
            preview = chunk.get('text', '')[:100] + '...'
            logger.info(f"Chunk {i}: Distance: {chunk['similarity_score']:.4f}, Preview: {preview}")
        
        return chunks[:k]  # Return top k results
        
    except Exception as e:
        logger.error(f"Error retrieving from knowledge base: {str(e)}")
        return []

def format_results(chunks):
    """Format retrieved chunks for display."""
    if not chunks:
        return "No relevant information found."
    
    result = []
    for i, chunk in enumerate(chunks, 1):
        result.append(f"\n--- Result {i} (Similarity: {chunk['similarity_score']:.4f}) ---")
        result.append(chunk.get('text', ''))
    
    return "\n".join(result)
