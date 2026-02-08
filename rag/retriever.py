import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ============================================
# CONFIGURATION
# ============================================

PROCESSED_DIR = os.path.join("data", "processed")
VECTOR_STORE_DIR = os.path.join(PROCESSED_DIR, "vector_store")
FAISS_INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
DOCUMENTS_FILE = os.path.join(VECTOR_STORE_DIR, "documents.json")

# Embedding model (same as used for creating embeddings)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================
# RETRIEVER CLASS
# ============================================

class ComplaintRetriever:
    """
    Retriever for customer complaints using FAISS vector store
    """
    
    def __init__(self):
        """Initialize retriever"""
        self.index = None
        self.documents = None
        self.embedding_model = None
        
    def load(self):
        """Load FAISS index, documents, and embedding model"""
        print("📥 Loading retriever components...")
        
        # Load FAISS index
        if not os.path.exists(FAISS_INDEX_FILE):
            raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_FILE}")
        
        self.index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load documents
        if not os.path.exists(DOCUMENTS_FILE):
            raise FileNotFoundError(f"Documents not found: {DOCUMENTS_FILE}")
        
        with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        print(f"✅ Loaded {len(self.documents)} documents")
        
        # Load embedding model
        print(f"🤖 Loading embedding model: {EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Embedding model loaded")
        
        return self
    
    def embed_query(self, query):
        """Convert query text to embedding"""
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not loaded. Call load() first.")
        
        embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        return embedding.astype('float32')
    
    def retrieve(self, query, top_k=5, return_scores=True):
        """
        Retrieve top-k most relevant documents for a query
        
        Args:
            query (str): User query
            top_k (int): Number of results to return
            return_scores (bool): Whether to return similarity scores
        
        Returns:
            list: Retrieved documents with optional scores
        """
        if self.index is None or self.documents is None:
            raise RuntimeError("Retriever not loaded. Call load() first.")
        
        # Embed query
        query_embedding = self.embed_query(query)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            doc = self.documents[idx].copy()
            
            if return_scores:
                # Convert L2 distance to similarity score (lower distance = higher similarity)
                # Using negative distance as score (higher is better)
                doc['score'] = float(-dist)
                doc['distance'] = float(dist)
                doc['rank'] = i + 1
            
            results.append(doc)
        
        return results
    
    def retrieve_with_context(self, query, top_k=5):
        """
        Retrieve documents and format with context
        
        Returns formatted context string suitable for LLM
        """
        results = self.retrieve(query, top_k=top_k, return_scores=True)
        
        # Format context
        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"[Document {i}]")
            context_parts.append(f"Source: {doc['source']}")
            context_parts.append(f"Text: {doc['text']}")
            context_parts.append(f"Relevance Score: {doc['score']:.4f}")
            context_parts.append("")  # Empty line
        
        context = "\n".join(context_parts)
        return context, results

# ============================================
# HELPER FUNCTIONS
# ============================================

def display_results(query, results):
    """Display retrieval results in a formatted way"""
    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)
    
    for i, doc in enumerate(results, 1):
        print(f"\n[Result {i}]")
        print(f"ID: {doc['id']}")
        print(f"Source: {doc['source']}")
        if 'score' in doc:
            print(f"Score: {doc['score']:.4f}")
            print(f"Distance: {doc['distance']:.4f}")
        print(f"Text: {doc['text'][:200]}...")
        print("-" * 60)

# ============================================
# TEST RETRIEVER
# ============================================

def test_retriever():
    """Test the retriever with sample queries"""
    print("="*60)
    print("  TESTING RETRIEVER")
    print("="*60)
    
    # Initialize and load retriever
    retriever = ComplaintRetriever()
    retriever.load()
    
    # Test queries
    test_queries = [
        "Product arrived damaged",
        "Late delivery issue",
        "Wrong item received",
        "Refund not processed",
        "Customer service problem"
    ]
    
    print("\n" + "="*60)
    print("RUNNING TEST QUERIES")
    print("="*60)
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        results = retriever.retrieve(query, top_k=3)
        display_results(query, results)
    
    # Test context retrieval
    print("\n" + "="*60)
    print("TEST CONTEXT RETRIEVAL")
    print("="*60)
    
    test_query = "Product quality complaint"
    context, results = retriever.retrieve_with_context(test_query, top_k=3)
    
    print(f"\nQuery: {test_query}")
    print("\nFormatted Context for LLM:")
    print("-" * 60)
    print(context[:500] + "..." if len(context) > 500 else context)
    
    print("\n✅ Retriever test complete!")

# ============================================
# MAIN
# ============================================

def main():
    """Main function"""
    test_retriever()

if __name__ == "__main__":
    main()