# Testing Guide: FAISS vs ChromaDB

## 📁 New Files Created

Your project now has both FAISS and ChromaDB implementations running side-by-side:

### FAISS Version (Original):
- `knowledge_base.py` - Creates FAISS index
- `retrieval.py` - Queries FAISS index
- `rag-chatbot.py` - FAISS-based chatbot
- `Master Compliance BMR_faiss.index` - FAISS index file
- `Master Compliance BMR_metadata.pkl` - Metadata file

### ChromaDB Version (New):
- `knowledge_base_chroma.py` - Creates ChromaDB collection
- `retrieval_chroma.py` - Queries ChromaDB collection
- `rag-chatbot-chroma.py` - ChromaDB-based chatbot
- `chroma_db/` - ChromaDB database folder (created after running)

### Comparison Tool:
- `compare_databases.py` - Side-by-side comparison of both databases

---

## 🚀 Step-by-Step Testing Guide

### Step 1: Install ChromaDB

```bash
pip install -r requirement.txt
```

This will install ChromaDB along with all other dependencies.

---

### Step 2: Build ChromaDB Database

Run the ChromaDB knowledge base builder:

```bash
python knowledge_base_chroma.py
```

**What this does:**
- Reads `Master Compliance BMR.pdf`
- Chunks the text (same as FAISS version)
- Generates embeddings using same model
- Creates ChromaDB collection at `./chroma_db/`
- Stores documents, embeddings, and metadata

**Expected output:**
```
Loading embedding model...
Processing file: Master Compliance BMR.pdf
Extracting text from PDF...
Split document into 181 chunks.
Generating embeddings...
Creating ChromaDB collection...
✓ ChromaDB collection created successfully!
✓ Total documents in collection: 181
```

---

### Step 3: Test ChromaDB Retrieval

Test the ChromaDB retrieval directly:

```bash
python retrieval_chroma.py
```

This runs test queries (pH value, temperature range, mixing time) against ChromaDB.

---

### Step 4: Test ChromaDB Chatbot

Run the ChromaDB-powered chatbot:

```bash
python rag-chatbot-chroma.py
```

**Usage:**
- Type any parameter name to search
- Type `raw` to toggle between LLM answers and raw chunks
- Type `quit` to exit

**Example:**
```
📋 Enter parameter name: pH value
🔍 Searching ChromaDB for: pH value
📖 Answer:
The pH value specification is 6.5-7.5...
```

---

### Step 5: Compare Both Databases

#### Option A: Interactive Comparison

```bash
python compare_databases.py
```

Then type queries to see side-by-side comparison.

#### Option B: Run Full Benchmark

```bash
python compare_databases.py --benchmark
```

This tests 5 predefined queries and shows:
- Query times for both databases
- Number of results returned
- Preview of top results
- Performance comparison
- Summary statistics

#### Option C: Test Specific Query

```bash
python compare_databases.py --query "pH value"
```

---

## 📊 What to Compare

### 1. Performance (Speed)
- Which database retrieves results faster?
- Average query time
- Total time for multiple queries

### 2. Result Quality
- Do both return similar results?
- Are the top results the same?
- Are similarity scores comparable?

### 3. Ease of Use
- Which code is simpler?
- File management (FAISS: 2 files vs ChromaDB: 1 folder)
- Query API simplicity

### 4. Features
- FAISS: Faster, lightweight
- ChromaDB: Updates, deletes, metadata filtering

---

## 🔍 Side-by-Side Comparison

### Query a Parameter

**FAISS Version:**
```bash
python rag-chatbot.py
📋 Enter parameter name: temperature
```

**ChromaDB Version:**
```bash
python rag-chatbot-chroma.py
📋 Enter parameter name: temperature
```

Compare the answers from both!

---

## 📈 Expected Results

### Performance:
- **FAISS**: Slightly faster (1-5ms advantage)
- **ChromaDB**: Very close, negligible difference for your dataset size

### Accuracy:
- Both should return similar/identical results
- Both use same embeddings and L2 distance
- Slight variations possible due to internal algorithms

### Storage:
- **FAISS**: ~2 files (index + pickle)
- **ChromaDB**: 1 folder with SQLite database

---

## 🧪 Test Scenarios

### Test 1: Basic Retrieval
```bash
python compare_databases.py --query "pH value"
```
Compare results and timing.

### Test 2: Multiple Queries
```bash
python compare_databases.py --benchmark
```
See average performance over 5 queries.

### Test 3: Complex Queries
Test with longer, more specific queries:
- "What is the acceptable range for mixing time?"
- "Storage conditions and temperature requirements"
- "Quality control parameters for batch release"

### Test 4: End-to-End RAG
Run both chatbots with same queries and compare final LLM answers.

---

## 📝 Evaluation Checklist

Use this to evaluate both databases:

- [ ] **Installation**: Was ChromaDB easy to install?
- [ ] **Database Creation**: Did ChromaDB build successfully?
- [ ] **Query Speed**: How does ChromaDB compare to FAISS?
- [ ] **Result Quality**: Do results match FAISS?
- [ ] **Code Simplicity**: Is ChromaDB code cleaner?
- [ ] **File Management**: Prefer single folder vs multiple files?
- [ ] **Documentation**: Is ChromaDB easier to understand?
- [ ] **Future Features**: Do you need updates/deletes?

---

## 🔄 Switching Between Databases

### Use FAISS:
```python
from retrieval import retrieve_from_knowledge_base
```

### Use ChromaDB:
```python
from retrieval_chroma import retrieve_from_knowledge_base
```

Both have the **same function signature**, so you can easily swap!

---

## 💾 Storage Comparison

### FAISS Files:
```
Master Compliance BMR_faiss.index    (~5-10 MB)
Master Compliance BMR_metadata.pkl   (~1-2 MB)
```

### ChromaDB Files:
```
chroma_db/
  └── [SQLite database and index files] (~10-15 MB)
```

ChromaDB uses slightly more storage but provides more features.

---

## 🎯 Key Differences You'll Notice

### FAISS:
✅ Slightly faster queries  
✅ Smaller file size  
✅ Lightweight dependencies  
❌ Manual file management  
❌ No updates without rebuilding  
❌ Separate metadata storage  

### ChromaDB:
✅ Simpler API  
✅ Automatic persistence  
✅ Can update/delete documents  
✅ Built-in metadata filtering  
✅ Better for production  
❌ Slightly slower (negligible)  
❌ Larger storage footprint  

---

## 🏆 Making Your Decision

After testing, consider:

1. **Performance**: Is the speed difference meaningful for your use case?
2. **Maintenance**: Which is easier to manage?
3. **Features**: Do you need document updates?
4. **Scale**: Planning to scale beyond current dataset?
5. **Team**: Which will your team find easier to work with?

For your current dataset size (~181 chunks), **performance will be nearly identical**. Choose based on ease of use and future needs.

---

## 🐛 Troubleshooting

### ChromaDB Not Found
```bash
pip install chromadb>=0.4.18
```

### Database Path Issues
Both databases use relative paths:
- FAISS: Current directory
- ChromaDB: `./chroma_db/`

Make sure to run scripts from the project root.

### Results Differ
Small differences are normal due to:
- Floating-point precision
- Internal algorithm variations
- Tie-breaking in similarity scores

---

## 📞 Next Steps

1. ✅ Build ChromaDB database: `python knowledge_base_chroma.py`
2. ✅ Test retrieval: `python retrieval_chroma.py`
3. ✅ Test chatbot: `python rag-chatbot-chroma.py`
4. ✅ Run comparison: `python compare_databases.py --benchmark`
5. ✅ Evaluate results
6. ✅ Choose your preferred database!

---

## 💡 Pro Tips

1. **Keep both implementations** during evaluation period
2. **Test with your specific queries** (not just examples)
3. **Measure what matters** to your use case
4. **Consider future needs**, not just current performance
5. **Document your findings** for future reference

---

**Happy Testing! 🚀**
