import json
import os
from rag.retriever import ComplaintRetriever
from rag.llm import HuggingFaceLLM


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
        self.llm = HuggingFaceLLM()

        
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
    context=context
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
# STREAMLIT WRAPPER FUNCTION
# ============================================

_pipeline_instance = None

def run_pipeline(query):
    global _pipeline_instance

    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline().load()

    result = _pipeline_instance.query(query, verbose=False)
    return result["answer"]
    