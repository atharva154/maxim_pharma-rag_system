# Maxim Pharma - RAG System

A Retrieval-Augmented Generation (RAG) chatbot system for pharmaceutical compliance documentation. This system uses vector databases (FAISS and ChromaDB) to efficiently retrieve and answer questions based on the Master Compliance BMR documentation.

## 🚀 Features

- **Dual Vector Database Support**: Implements both FAISS and ChromaDB for flexible document retrieval
- **PDF-based Knowledge Extraction**: Automatically processes pharmaceutical compliance documents
- **Intelligent Question Answering**: Uses LLM (OpenAI/Groq) to provide accurate, context-aware responses
- **Semantic Search**: Retrieves the most relevant document chunks based on query similarity
- **Comparison Tools**: Compare performance between different vector database implementations

## 📁 Project Structure

```
RAG-Chatbot/
├── rag-chatbot.py              # Main chatbot implementation (FAISS)
├── rag-chatbot-chroma.py       # Main chatbot implementation (ChromaDB)
├── knowledge_base.py           # FAISS knowledge base creation
├── knowledge_base_chroma.py    # ChromaDB knowledge base creation
├── retrieval.py                # FAISS retrieval logic
├── retrieval_chroma.py         # ChromaDB retrieval logic
├── config_chroma.py            # ChromaDB configuration
├── compare_databases.py        # Database comparison utility
├── Master Compliance BMR_faiss.index  # FAISS index file
├── chroma_db/                  # ChromaDB storage
├── .env                        # Environment variables (API keys)
├── requirement.txt             # Python dependencies
├── QUICK_REFERENCE.md          # Quick reference guide
└── TESTING_GUIDE.md            # Testing documentation
```

## 🛠️ Technologies Used

- **Python 3.x**
- **LangChain**: Framework for LLM applications
- **FAISS**: Facebook AI Similarity Search for vector storage
- **ChromaDB**: Open-source embedding database
- **OpenAI API**: For embeddings and LLM capabilities
- **Groq API**: Alternative LLM provider
- **PyPDF2/PDFPlumber**: PDF processing

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Groq API key (optional)

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/atharva154/maxim_pharma-rag_system.git
   cd maxim_pharma-rag_system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🚀 Usage

### Using FAISS Implementation

1. **Create Knowledge Base**
   ```bash
   python knowledge_base.py
   ```

2. **Run the Chatbot**
   ```bash
   python rag-chatbot.py
   ```

### Using ChromaDB Implementation

1. **Create Knowledge Base**
   ```bash
   python knowledge_base_chroma.py
   ```

2. **Run the Chatbot**
   ```bash
   python rag-chatbot-chroma.py
   ```

### Compare Database Performance

```bash
python compare_databases.py
```

## 💡 How It Works

1. **Document Processing**: PDF documents are loaded and split into manageable chunks
2. **Embedding Generation**: Text chunks are converted into vector embeddings using OpenAI's embedding model
3. **Vector Storage**: Embeddings are stored in FAISS or ChromaDB for efficient similarity search
4. **Query Processing**: User questions are embedded and matched against stored vectors
5. **Context Retrieval**: Most relevant document chunks are retrieved
6. **Response Generation**: LLM generates accurate answers based on retrieved context

## 📖 Documentation

- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick start guide and common operations
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing procedures and validation

## 🔒 Security

- Never commit your `.env` file to version control
- Keep your API keys secure and rotate them regularly
- The `.gitignore` file is configured to exclude sensitive files

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/<Feature-name>`)
3. Commit your changes (`git commit -m 'Add some Feature'`)
4. Push to the branch (`git push origin feature/<Feature-name>`)
5. Open a Pull Request
