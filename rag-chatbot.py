import os
from groq import Groq
from dotenv import load_dotenv
from retrieval import retrieve_from_knowledge_base, format_results
import sys

# Load environment variables
load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not found in .env file")
    sys.exit(1)

client = Groq(api_key=api_key)

def answer_query(parameter_name, use_llm=True):
    """
    Answer query about a parameter using RAG.
    
    Args:
        parameter_name (str): Name of the parameter to search for
        use_llm (bool): Whether to use LLM for answer generation or just return raw chunks
        
    Returns:
        str: Answer or retrieved context
    """
    print(f"\n🔍 Searching for: {parameter_name}")
    
    # Retrieve relevant context
    chunks = retrieve_from_knowledge_base(parameter_name, k=3)
    
    if not chunks:
        return "❌ No relevant information found in knowledge base."
    
    # Option 1: Just return formatted chunks (no LLM)
    if not use_llm:
        return format_results(chunks)
    
    # Option 2: Use LLM to generate answer from context
    context = "\n\n".join([chunk['text'] for chunk in chunks])
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are a pharmaceutical BMR expert. Answer questions based ONLY on the provided context from Master BMR.

Rules:
- Extract exact values and specifications
- If the parameter has a range, provide the full range
- If the parameter has units, include them
- Be concise and precise
- If not found in context, say "Not specified in provided context"
"""
                },
                {
                    "role": "user",
                    "content": f"Parameter: {parameter_name}\n\nContext from Master BMR:\n{context}\n\nProvide the value, specifications, and requirements for this parameter."
                }
            ],
            temperature=0,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

def interactive_mode():
    """Run chatbot in interactive CLI mode."""
    print("=" * 70)
    print("🤖 RAG Chatbot - BMR Parameter Lookup")
    print("=" * 70)
    print("\nCommands:")
    print("  • Type parameter name to search")
    print("  • Type 'raw' to toggle raw/LLM mode")
    print("  • Type 'quit' or 'exit' to quit")
    print("=" * 70)
    
    use_llm = True
    
    while True:
        try:
            query = input("\n📋 Enter parameter name: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'raw':
                use_llm = not use_llm
                mode = "LLM-generated answers" if use_llm else "Raw chunks"
                print(f"✅ Switched to: {mode}")
                continue
            
            # Get answer
            answer = answer_query(query, use_llm=use_llm)
            
            # Display answer
            print("\n" + "=" * 70)
            print("📄 Answer:")
            print("=" * 70)
            print(answer)
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

def batch_mode(parameters):
    """Process multiple parameters in batch."""
    print(f"\n📊 Processing {len(parameters)} parameters...\n")
    
    results = {}
    for param in parameters:
        print(f"Processing: {param}")
        answer = answer_query(param, use_llm=True)
        results[param] = answer
    
    return results

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        # Command line mode: python chatbot.py "Batch Size"
        param = " ".join(sys.argv[1:])
        answer = answer_query(param, use_llm=True)
        print("\n" + "=" * 70)
        print("📄 Answer:")
        print("=" * 70)
        print(answer)
        print("=" * 70)
    else:
        # Interactive mode
        interactive_mode()
