import os
import hashlib
import time
import pickle
import numpy as np
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
print(f"Embedding model loaded. Dimension: {embedding_model.get_sentence_embedding_dimension()}")

# Load environment variables
load_dotenv()

# Chunking Configuration
CHUNK_SIZE = 300  # Number of words per chunk
CHUNK_OVERLAP = 50  # Number of overlapping words

# Batch size for embedding generation
BATCH_SIZE = 32  # Process embeddings in batches

print(f"Using local embedding model: Qwen/Qwen3-Embedding-0.6B")

# --- 3. Hardcoded Input File Path ---
INPUT_FILE_PATH = "Master Compliance BMR.pdf"

# --- 4. Helper Functions ---

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

# --- 5. Main Creation Logic ---

def create_database():
    """Main function to create the FAISS database from the input text file."""
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
    for i, chunk in enumerate(chunks_text):
        meta = {
            "source": os.path.basename(input_filepath),
            "chunk_id": generate_chunk_id(chunk),
            "chunk_index": i,
            "text": chunk  # Store the actual text in metadata for later retrieval
        }
        all_metadata.append(meta)

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
    embeddings_np = all_embeddings.astype('float32')
    print("Embeddings generated successfully.")

    # Step 4: Create and Save FAISS Index
    dimension = embeddings_np.shape[1]
    # Use IndexFlatL2 for L2 distance (Euclidean distance)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    # Generate output filenames
    base_filename = os.path.splitext(os.path.basename(input_filepath))[0]
    index_file = f"{base_filename}_faiss.index"
    metadata_file = f"{base_filename}_metadata.pkl"
    
    faiss.write_index(index, index_file)
    print(f"\nFAISS index created with {index.ntotal} vectors of dimension {dimension}.")
    print(f"Index saved to: {index_file}")

    # Step 5: Save Metadata
    with open(metadata_file, 'wb') as f:
        pickle.dump(all_metadata, f)
    print(f"Metadata saved to: {metadata_file}")

    print("\nDatabase creation process complete!")

if __name__ == "__main__":
    create_database()
