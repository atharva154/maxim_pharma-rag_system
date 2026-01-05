"""
Script to compare FAISS and ChromaDB performance and results side-by-side.
"""
import time
from sentence_transformers import SentenceTransformer

# Import both retrieval modules
import retrieval  # FAISS version
import retrieval_chroma  # ChromaDB version

# Test queries
TEST_QUERIES = [
    "pH value",
    "temperature range",
    "mixing time",
    "batch size",
    "storage conditions"
]

def compare_single_query(query, k=5):
    """Compare FAISS and ChromaDB results for a single query."""
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print('='*80)
    
    # Test FAISS
    print("\n[FAISS] Searching...")
    start_time = time.time()
    faiss_results = retrieval.retrieve_from_knowledge_base(query, k=k)
    faiss_time = time.time() - start_time
    
    # Test ChromaDB
    print("\n[ChromaDB] Searching...")
    start_time = time.time()
    chroma_results = retrieval_chroma.retrieve_from_knowledge_base(query, k=k)
    chroma_time = time.time() - start_time
    
    # Display comparison
    print(f"\n{'─'*80}")
    print("COMPARISON RESULTS")
    print('─'*80)
    print(f"Query: '{query}'")
    print(f"\nFAISS:")
    print(f"  - Results found: {len(faiss_results)}")
    print(f"  - Query time: {faiss_time*1000:.2f}ms")
    
    print(f"\nChromaDB:")
    print(f"  - Results found: {len(chroma_results)}")
    print(f"  - Query time: {chroma_time*1000:.2f}ms")
    
    # Performance comparison
    if faiss_time < chroma_time:
        faster = "FAISS"
        diff = ((chroma_time - faiss_time) / faiss_time) * 100
    else:
        faster = "ChromaDB"
        diff = ((faiss_time - chroma_time) / chroma_time) * 100
    
    print(f"\nPerformance: {faster} is {diff:.1f}% faster")
    
    # Show top results side by side
    print(f"\n{'─'*80}")
    print("TOP 3 RESULTS COMPARISON")
    print('─'*80)
    
    max_results = min(3, len(faiss_results), len(chroma_results))
    
    for i in range(max_results):
        print(f"\n--- Result #{i+1} ---")
        
        # FAISS result
        if i < len(faiss_results):
            faiss_chunk = faiss_results[i]
            print(f"\n[FAISS] Score: {faiss_chunk['similarity_score']:.4f}")
            print(f"Preview: {faiss_chunk['text'][:150]}...")
        
        # ChromaDB result
        if i < len(chroma_results):
            chroma_chunk = chroma_results[i]
            print(f"\n[ChromaDB] Distance: {chroma_chunk['similarity_score']:.4f}")
            print(f"Preview: {chroma_chunk['text'][:150]}...")
    
    return {
        'query': query,
        'faiss_time': faiss_time,
        'chroma_time': chroma_time,
        'faiss_count': len(faiss_results),
        'chroma_count': len(chroma_results)
    }

def run_benchmark():
    """Run comprehensive benchmark comparing both databases."""
    print("\n" + "="*80)
    print("FAISS vs ChromaDB Benchmark")
    print("="*80)
    print(f"\nTesting {len(TEST_QUERIES)} queries...")
    print(f"Retrieving top 5 results per query")
    
    results = []
    
    for query in TEST_QUERIES:
        result = compare_single_query(query, k=5)
        results.append(result)
        time.sleep(0.5)  # Small delay between tests
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print('='*80)
    
    total_faiss_time = sum(r['faiss_time'] for r in results)
    total_chroma_time = sum(r['chroma_time'] for r in results)
    
    avg_faiss_time = total_faiss_time / len(results)
    avg_chroma_time = total_chroma_time / len(results)
    
    print(f"\nAverage Query Times:")
    print(f"  FAISS:    {avg_faiss_time*1000:.2f}ms")
    print(f"  ChromaDB: {avg_chroma_time*1000:.2f}ms")
    
    print(f"\nTotal Query Times:")
    print(f"  FAISS:    {total_faiss_time*1000:.2f}ms")
    print(f"  ChromaDB: {total_chroma_time*1000:.2f}ms")
    
    # Determine overall winner
    if avg_faiss_time < avg_chroma_time:
        faster = "FAISS"
        diff = ((avg_chroma_time - avg_faiss_time) / avg_faiss_time) * 100
    else:
        faster = "ChromaDB"
        diff = ((avg_faiss_time - avg_chroma_time) / avg_chroma_time) * 100
    
    print(f"\n🏆 Overall: {faster} is {diff:.1f}% faster on average")
    
    # Detailed results table
    print(f"\n{'─'*80}")
    print("DETAILED RESULTS")
    print('─'*80)
    print(f"{'Query':<25} {'FAISS (ms)':<15} {'ChromaDB (ms)':<15} {'Winner':<10}")
    print('─'*80)
    
    for r in results:
        faiss_ms = r['faiss_time'] * 1000
        chroma_ms = r['chroma_time'] * 1000
        winner = "FAISS" if faiss_ms < chroma_ms else "ChromaDB"
        
        print(f"{r['query']:<25} {faiss_ms:<15.2f} {chroma_ms:<15.2f} {winner:<10}")
    
    print('='*80)

def compare_specific_query():
    """Interactive mode to test specific queries."""
    print("\n" + "="*80)
    print("Interactive Query Comparison")
    print("="*80)
    print("\nCommands:")
    print("  • Type a query to compare both databases")
    print("  • Type 'benchmark' to run full benchmark")
    print("  • Type 'quit' or 'exit' to quit")
    print("="*80)
    
    while True:
        try:
            query = input("\n📋 Enter query (or command): ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'benchmark':
                run_benchmark()
                continue
            
            # Compare the query
            compare_single_query(query, k=5)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--benchmark':
            run_benchmark()
        elif sys.argv[1] == '--query' and len(sys.argv) > 2:
            query = ' '.join(sys.argv[2:])
            compare_single_query(query, k=5)
        else:
            print("Usage:")
            print("  python compare_databases.py              # Interactive mode")
            print("  python compare_databases.py --benchmark  # Run full benchmark")
            print("  python compare_databases.py --query <query>  # Test specific query")
    else:
        # Interactive mode
        compare_specific_query()
