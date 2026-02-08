import json
import os
import pickle
import numpy as np
import faiss

# ============================================
# CONFIGURATION
# ============================================

PROCESSED_DIR = os.path.join("data", "processed")
EMBEDDINGS_FILE = os.path.join(PROCESSED_DIR, "embeddings.pkl")
DOCUMENTS_FILE = os.path.join(PROCESSED_DIR, "documents_with_embeddings.json")
VECTOR_STORE_DIR = os.path.join(PROCESSED_DIR, "vector_store")
FAISS_INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
DOCUMENTS_STORE_FILE = os.path.join(VECTOR_STORE_DIR, "documents.json")

# ============================================
# STEP 1: LOAD EMBEDDINGS AND DOCUMENTS
# ============================================

def load_embeddings_and_documents():
    """Load embeddings and documents from files"""
    print("\n📥 Loading embeddings and documents...")
    
    # Load embeddings
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"❌ Error: {EMBEDDINGS_FILE} not found!")
        print("   Please run rag/create_embeddings.py first.")
        return None, None
    
    with open(EMBEDDINGS_FILE, 'rb') as f:
        embeddings = pickle.load(f)
    
    print(f"✅ Loaded embeddings: {embeddings.shape}")
    
    # Load documents
    if not os.path.exists(DOCUMENTS_FILE):
        print(f"❌ Error: {DOCUMENTS_FILE} not found!")
        return None, None
    
    with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"✅ Loaded documents: {len(documents)}")
    
    # Verify match
    if len(embeddings) != len(documents):
        print(f"⚠️  Warning: Mismatch between embeddings ({len(embeddings)}) and documents ({len(documents)})")
        return None, None
    
    return embeddings, documents

# ============================================
# STEP 2: CREATE FAISS INDEX
# ============================================

def create_faiss_index(embeddings):
    """
    Create FAISS index for fast similarity search
    Using IndexFlatL2 for exact search (good for small-medium datasets)
    """
    print("\n🔧 Creating FAISS index...")
    
    # Get embedding dimension
    dimension = embeddings.shape[1]
    print(f"   Embedding dimension: {dimension}")
    print(f"   Number of vectors: {len(embeddings)}")
    
    # Create FAISS index
    # IndexFlatL2 = exact search using L2 distance (Euclidean)
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to index
    print("   Adding embeddings to index...")
    index.add(embeddings.astype('float32'))
    
    print(f"✅ FAISS index created!")
    print(f"   Total vectors in index: {index.ntotal}")
    
    return index

# ============================================
# STEP 3: TEST VECTOR STORE
# ============================================

def test_vector_store(index, embeddings, documents):
    """Test the vector store with a sample query"""
    print("\n🧪 Testing vector store...")
    
    # Use first document as test query
    test_embedding = embeddings[0:1].astype('float32')
    test_doc = documents[0]
    
    print(f"\nTest query document:")
    print(f"   ID: {test_doc['id']}")
    print(f"   Text: {test_doc['text'][:100]}...")
    
    # Search for top 5 similar documents
    k = 5
    distances, indices = index.search(test_embedding, k)
    
    print(f"\n🔍 Top {k} similar documents:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        print(f"\n   {i+1}. Document ID: {documents[idx]['id']}")
        print(f"      Distance: {dist:.4f}")
        print(f"      Text: {documents[idx]['text'][:80]}...")
    
    print("\n✅ Vector store test passed!")

# ============================================
# STEP 4: SAVE VECTOR STORE
# ============================================

def save_vector_store(index, documents):
    """Save FAISS index and documents"""
    print("\n💾 Saving vector store...")
    
    # Create directory
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"✅ FAISS index saved to: {FAISS_INDEX_FILE}")
    print(f"   File size: {os.path.getsize(FAISS_INDEX_FILE) / 1024:.2f} KB")
    
    # Save documents
    with open(DOCUMENTS_STORE_FILE, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Documents saved to: {DOCUMENTS_STORE_FILE}")
    print(f"   File size: {os.path.getsize(DOCUMENTS_STORE_FILE) / 1024:.2f} KB")
    
    return FAISS_INDEX_FILE, DOCUMENTS_STORE_FILE

# ============================================
# STEP 5: CREATE RETRIEVER HELPER
# ============================================

def create_retriever_info():
    """Create info file for easy loading"""
    info = {
        "vector_store_type": "FAISS",
        "index_file": FAISS_INDEX_FILE,
        "documents_file": DOCUMENTS_STORE_FILE,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "total_documents": None  # Will be filled
    }
    
    info_file = os.path.join(VECTOR_STORE_DIR, "store_info.json")
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Store info saved to: {info_file}")
    return info_file

# ============================================
# MAIN PIPELINE
# ============================================

def main():
    """Main vector store creation pipeline"""
    
    print("="*60)
    print("  FAISS VECTOR STORE CREATION")
    print("="*60)
    
    # Step 1: Load data
    print("\n" + "="*60)
    print("STEP 1: LOAD EMBEDDINGS AND DOCUMENTS")
    print("="*60)
    embeddings, documents = load_embeddings_and_documents()
    if embeddings is None or documents is None:
        return
    
    # Step 2: Create FAISS index
    print("\n" + "="*60)
    print("STEP 2: CREATE FAISS INDEX")
    print("="*60)
    index = create_faiss_index(embeddings)
    
    # Step 3: Test vector store
    print("\n" + "="*60)
    print("STEP 3: TEST VECTOR STORE")
    print("="*60)
    test_vector_store(index, embeddings, documents)
    
    # Step 4: Save vector store
    print("\n" + "="*60)
    print("STEP 4: SAVE VECTOR STORE")
    print("="*60)
    index_file, docs_file = save_vector_store(index, documents)
    
    # Step 5: Create info file
    print("\n" + "="*60)
    print("STEP 5: CREATE RETRIEVER INFO")
    print("="*60)
    info_file = create_retriever_info()
    
    # Final summary
    print("\n" + "="*60)
    print("✅ VECTOR STORE CREATION COMPLETE!")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   - Total documents indexed: {len(documents)}")
    print(f"   - Embedding dimension: {embeddings.shape[1]}")
    print(f"   - Index type: FAISS IndexFlatL2 (exact search)")
    print(f"\n📁 Output files:")
    print(f"   - FAISS index: {index_file}")
    print(f"   - Documents: {docs_file}")
    print(f"   - Store info: {info_file}")
    print(f"\n🎯 Ready for retrieval and RAG pipeline!")

if __name__ == "__main__":
    main()