# Quick Reference: FAISS vs ChromaDB

## 📋 Command Cheatsheet

### Setup
```bash
# Install dependencies (includes ChromaDB)
pip install -r requirement.txt
```

### Build Databases
```bash
# Build FAISS database (already done)
python knowledge_base.py

# Build ChromaDB database (new)
python knowledge_base_chroma.py
```

### Test Retrieval
```bash
# Test FAISS retrieval
python retrieval.py

# Test ChromaDB retrieval
python retrieval_chroma.py
```

### Run Chatbots
```bash
# FAISS-powered chatbot
python rag-chatbot.py

# ChromaDB-powered chatbot
python rag-chatbot-chroma.py
```

### Compare Both
```bash
# Interactive comparison
python compare_databases.py

# Full benchmark (5 test queries)
python compare_databases.py --benchmark

# Test specific query
python compare_databases.py --query "pH value"
```

---

## 📂 File Structure

```
RAG-Chatbot/
├── FAISS Implementation:
│   ├── knowledge_base.py
│   ├── retrieval.py
│   ├── rag-chatbot.py
│   ├── Master Compliance BMR_faiss.index
│   └── Master Compliance BMR_metadata.pkl
│
├── ChromaDB Implementation:
│   ├── knowledge_base_chroma.py
│   ├── retrieval_chroma.py
│   ├── rag-chatbot-chroma.py
│   └── chroma_db/  (folder, created after running)
│
├── Comparison & Guides:
│   ├── compare_databases.py
│   ├── TESTING_GUIDE.md
│   └── MIGRATION_TO_CHROMADB.md
│
└── Shared Files:
    ├── requirement.txt
    ├── .env
    └── Master Compliance BMR.pdf
```

---

## 🔄 Function Signatures (Identical!)

Both implementations have the same API:

```python
# Both work the same way!
from retrieval import retrieve_from_knowledge_base  # FAISS
from retrieval_chroma import retrieve_from_knowledge_base  # ChromaDB

# Usage is identical
results = retrieve_from_knowledge_base(
    query="pH value",
    k=5  # number of results
)
```

---

## ⚡ Quick Test

Test both databases in 2 minutes:

```bash
# 1. Build ChromaDB (30 seconds)
python knowledge_base_chroma.py

# 2. Compare them (30 seconds)
python compare_databases.py --query "temperature"

# 3. See full benchmark (1 minute)
python compare_databases.py --benchmark
```

---

## 📊 What to Look For

When comparing results, check:

1. **Query Time**: Shown in milliseconds (ms)
2. **Result Count**: Number of chunks returned
3. **Similarity Scores**: Lower = more similar (L2 distance)
4. **Result Content**: Are top results the same?

---

## 🎯 Quick Decision Guide

**Choose FAISS if:**
- Need maximum speed at scale
- Working with 1M+ vectors
- File-based storage preferred
- Minimal dependencies

**Choose ChromaDB if:**
- Need to update/delete documents
- Want simpler API
- Building production app
- **For your use case: ✅ Recommended**

---

## 💡 Example Session

```bash
# Build ChromaDB
$ python knowledge_base_chroma.py
✓ ChromaDB collection created successfully!
✓ Total documents in collection: 181

# Compare query
$ python compare_databases.py --query "pH value"

Query: pH value
─────────────────────────────────
FAISS:
  - Results found: 3
  - Query time: 12.45ms

ChromaDB:
  - Results found: 3
  - Query time: 14.23ms

Performance: FAISS is 14.3% faster

🏆 Both returned similar results!
```

---

## 🚀 Next Steps

1. Run: `python knowledge_base_chroma.py`
2. Run: `python compare_databases.py --benchmark`
3. Review: Compare performance and results
4. Decide: Which database fits your needs?

**That's it! You now have both databases running side-by-side for testing.**
