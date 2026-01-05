import os
import hashlib
import time
import chromadb
from chromadb.config import Settings
import pdfplumber
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from config_chroma import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
    INSERT_BATCH_SIZE,
    INPUT_FILE_PATH
)

# Load environment variables
load_dotenv()

# Load embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"Embedding model loaded: {EMBEDDING_MODEL_NAME}")
print(f"Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")

# Display ChromaDB Configuration
print(f"\nChromaDB Configuration:")
print(f"  Database Path: {CHROMA_DB_PATH}")
print(f"  Collection Name: {COLLECTION_NAME}")
print(f"  Chunk Size: {CHUNK_SIZE} words")
print(f"  Chunk Overlap: {CHUNK_OVERLAP} words")

# --- Helper Functions ---

def read_text_from_file(file_path):
    """
    Reads and returns the content from a file (supports .txt and .pdf).
    Handles potential errors.
    """
    try:
        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            # Extract text from PDF
            print("Extracting text from PDF...")
            text_content = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                    if i % 10 == 0:
                        print(f"  Processed {i}/{len(pdf.pages)} pages...")
            
            full_text = '\n\n'.join(text_content)
            print(f"Successfully extracted text from {len(pdf.pages)} pages")
            return full_text
            
        elif file_ext == '.txt':
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            print(f"Error: Unsupported file format '{file_ext}'. Please use .txt or .pdf")
            return None
            
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def chunk_text_simple(text, chunk_size=300, overlap=50):
    """
    Simple text chunking by word count without external dependencies.
    
    Args:
        text (str): Text to chunk
        chunk_size (int): Number of words per chunk
        overlap (int): Number of overlapping words between chunks
        
    Returns:
        list: List of text chunks
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))
        
        if end >= len(words):
            break
            
        start += (chunk_size - overlap)
    
    return chunks

def generate_chunk_id(content):
    """Generates a unique MD5 hash ID for a chunk of text."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

# --- Main Creation Logic ---

def create_database():
    """Main function to create the ChromaDB database from the input text file."""
    input_filepath = INPUT_FILE_PATH
    print(f"\nProcessing file: {input_filepath}")

    # Step 1: Load and Chunk the Document
    raw_text = read_text_from_file(input_filepath)
    if raw_text is None:
        return  # Exit if file could not be read

    print("Text loaded successfully.")

    # Chunk the text using simple word-based chunking
    chunks_text = chunk_text_simple(raw_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    
    if not chunks_text:
        print("No text chunks were generated. The input file might be empty or too short.")
        return
        
    print(f"Split document into {len(chunks_text)} chunks.")

    # Step 2: Prepare Chunks and Metadata
    all_metadata = []
    all_ids = []
    
    for i, chunk in enumerate(chunks_text):
        chunk_id = generate_chunk_id(chunk)
        meta = {
            "source": os.path.basename(input_filepath),
            "chunk_id": chunk_id,
            "chunk_index": i
        }
        all_metadata.append(meta)
        all_ids.append(f"chunk_{i}_{chunk_id[:8]}")  # Unique ID

    # Step 3: Generate Embeddings using local model (batch processing)
    print(f"\nGenerating embeddings for {len(chunks_text)} chunks...")
    
    # Generate embeddings in batch (much faster than one-by-one)
    all_embeddings = embedding_model.encode(
        chunks_text,
        batch_size=BATCH_SIZE,  # Process 32 chunks at once
        show_progress_bar=True,  # Show progress
        convert_to_numpy=True
    )
    
    print(f"Generated embeddings with shape: {all_embeddings.shape}")
    embeddings_list = all_embeddings.astype('float32').tolist()  # Convert to list for ChromaDB
    print("Embeddings converted to list format.")

    # Step 4: Initialize ChromaDB and Create Collection
    print(f"\nInitializing ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Delete collection if it exists (for clean rebuild)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass
    
    # Create new collection with L2 distance (same as FAISS)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "l2"}  # Use L2 distance like FAISS
    )
    print(f"Created collection: {COLLECTION_NAME}")

    # Step 5: Add Documents to Collection (batch insert)
    print(f"\nAdding {len(chunks_text)} documents to ChromaDB...")
    
    # ChromaDB has a limit on batch size, so we'll insert in batches
    for i in range(0, len(chunks_text), INSERT_BATCH_SIZE):
        batch_end = min(i + INSERT_BATCH_SIZE, len(chunks_text))
        
        collection.add(
            documents=chunks_text[i:batch_end],
            embeddings=embeddings_list[i:batch_end],
            metadatas=all_metadata[i:batch_end],
            ids=all_ids[i:batch_end]
        )
        
        print(f"  Inserted batch {i//INSERT_BATCH_SIZE + 1}: documents {i} to {batch_end-1}")
    
    # Verify the collection
    total_docs = collection.count()
    print(f"\n✓ ChromaDB collection created successfully!")
    print(f"✓ Total documents in collection: {total_docs}")
    print(f"✓ Collection name: {COLLECTION_NAME}")
    print(f"✓ Database location: {CHROMA_DB_PATH}")
    print("\nDatabase creation process complete!")

if __name__ == "__main__":
    create_database()
