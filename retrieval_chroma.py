from sentence_transformers import SentenceTransformer
import chromadb
import logging
from config_chroma import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
print(f"Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")
print(f"Using ChromaDB collection: {COLLECTION_NAME}")

def retrieve_from_knowledge_base(query, k=5):
    """
    Retrieve relevant chunks from ChromaDB knowledge base for a given query.
    
    Args:
        query (str): Parameter name or question to search for
        k (int): Number of results to return
        
    Returns:
        list: Retrieved chunks with similarity scores
    """
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        
        logger.info(f"Connected to ChromaDB collection: {COLLECTION_NAME}")
        logger.info(f"Total documents in collection: {collection.count()}")
        
        # Generate query embedding
        query_embedding = embedding_model.encode(query, convert_to_numpy=True)
        
        # Query ChromaDB collection
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k * 2,  # Retrieve more candidates for filtering
            include=["documents", "metadatas", "distances"]
        )
        
        # Process and filter results
        chunks = []
        if results['ids'][0]:  # Check if we have results
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                
                # Filter by distance threshold (L2 distance)
                if distance < 2.0:
                    chunk = {
                        'text': results['documents'][0][i],
                        'similarity_score': float(distance),
                        'chunk_index': results['metadatas'][0][i].get('chunk_index'),
                        'source': results['metadatas'][0][i].get('source'),
                        'chunk_id': results['metadatas'][0][i].get('chunk_id'),
                        'id': results['ids'][0][i]
                    }
                    chunks.append(chunk)
        
        # Log results
        logger.info(f"Found {len(chunks)} relevant chunks for query: {query[:100]}...")
        for i, chunk in enumerate(chunks[:k], 1):
            preview = chunk.get('text', '')[:100] + '...'
            logger.info(f"Chunk {i}: Distance: {chunk['similarity_score']:.4f}, Preview: {preview}")
        
        return chunks[:k]  # Return top k results
        
    except Exception as e:
        logger.error(f"Error retrieving from ChromaDB: {str(e)}")
        return []

def format_results(chunks):
    """Format retrieved chunks for display."""
    if not chunks:
        return "No relevant information found."
    
    result = []
    for i, chunk in enumerate(chunks, 1):
        result.append(f"\n--- Result {i} (Distance: {chunk['similarity_score']:.4f}) ---")
        result.append(chunk.get('text', ''))
    
    return "\n".join(result)

# Test retrieval if run directly
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing ChromaDB Retrieval")
    print("="*70)
    
    # Test queries
    test_queries = [
        "pH value",
        "temperature range",
        "mixing time"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        
        chunks = retrieve_from_knowledge_base(query, k=3)
        print(format_results(chunks))
