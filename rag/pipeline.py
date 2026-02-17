import json
import os
from rag.retriever import ComplaintRetriever
from rag.llm import OllamaLLM

# ============================================
# RAG PIPELINE CLASS
# ============================================

class RAGPipeline:
    """
    Complete RAG Pipeline
    Combines Retriever + LLM to answer questions
    """
    
    def __init__(self):
        """Initialize RAG pipeline"""
        self.retriever = None
        self.llm = None
        
    def load(self):
        """Load retriever and LLM"""
        print("="*60)
        print("  INITIALIZING RAG PIPELINE")
        print("="*60)
        
        # Load retriever
        print("\n[1/2] Loading Retriever...")
        self.retriever = ComplaintRetriever()
        self.retriever.load()
        
        # Load LLM
        print("\n[2/2] Loading LLM...")
        self.llm = OllamaLLM()
        
        print("\n" + "="*60)
        print("✅ RAG PIPELINE READY!")
        print("="*60)
        
        return self
    
    def query(self, user_query, top_k=5, max_tokens=500, temperature=0.7, verbose=True):
        """
        Answer a user query using RAG
        
        Args:
            user_query (str): User's question
            top_k (int): Number of documents to retrieve
            max_tokens (int): Maximum tokens for LLM response
            temperature (float): LLM temperature
            verbose (bool): Print detailed info
        
        Returns:
            dict: Response with answer, sources, and metadata
        """
        if self.retriever is None or self.llm is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        
        if verbose:
            print("\n" + "="*60)
            print(f"QUERY: {user_query}")
            print("="*60)
        
        # Step 1: Retrieve relevant documents
        if verbose:
            print(f"\n🔍 Retrieving top {top_k} relevant documents...")
        
        context, retrieved_docs = self.retriever.retrieve_with_context(user_query, top_k=top_k)
        
        if verbose:
            print(f"✅ Retrieved {len(retrieved_docs)} documents")
            print("\nTop 3 sources:")
            for i, doc in enumerate(retrieved_docs[:3], 1):
                print(f"  {i}. {doc['source']} - Score: {doc.get('score', 'N/A'):.4f}")
                print(f"     Preview: {doc['text'][:80]}...")
        
        # Step 2: Generate answer using LLM
        if verbose:
            print(f"\n🤖 Generating answer using LLM...")
        
        answer = self.llm.generate_with_context(
            query=user_query,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if verbose:
            print("\n" + "="*60)
            print("ANSWER")
            print("="*60)
            print(answer)
            print("="*60)
        
        # Prepare response
        response = {
            'query': user_query,
            'answer': answer,
            'sources': retrieved_docs,
            'num_sources': len(retrieved_docs),
            'context_used': context
        }
        
        return response

# ============================================
# TEST RAG PIPELINE
# ============================================

def test_rag_pipeline():
    """Test the complete RAG pipeline"""
    print("="*60)
    print("  TESTING COMPLETE RAG PIPELINE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    pipeline.load()
    
    # Test queries
    test_queries = [
        "Why does the app crash when placing orders?",
        "Items disappearing from cart",
        "How do I get a refund?"
    ]
    
    print("\n" + "="*60)
    print("RUNNING TEST QUERIES")
    print("="*60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print(f"{'='*60}")
        
        response = pipeline.query(query, top_k=3, verbose=True)
        
        # Add some spacing
        print("\n")
    
    print("="*60)
    print("✅ RAG PIPELINE TEST COMPLETE!")
    print("="*60)

# ============================================
# STREAMLIT WRAPPER FUNCTION
# ============================================

_pipeline_instance = None

def run_pipeline(query):
    """
    Wrapper function for Streamlit
    Maintains a single pipeline instance
    """
    global _pipeline_instance

    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline().load()

    result = _pipeline_instance.query(query, verbose=False)
    return result["answer"]

# ============================================
# MAIN
# ============================================

def main():
    """Main function"""
    test_rag_pipeline()

if __name__ == "__main__":
    main()