import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

# ============================================
# CONFIGURATION
# ============================================

PROCESSED_DIR = os.path.join("data", "processed")
INPUT_FILE = os.path.join(PROCESSED_DIR, "cleaned_rag_documents.json")
EMBEDDINGS_FILE = os.path.join(PROCESSED_DIR, "embeddings.pkl")
DOCUMENTS_FILE = os.path.join(PROCESSED_DIR, "documents_with_embeddings.json")

# Embedding model - lightweight and fast
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast, good quality

# ============================================
# STEP 1: LOAD CLEANED DOCUMENTS
# ============================================

def load_cleaned_documents():
    """Load cleaned documents from JSON"""
    print("\n📥 Loading cleaned documents...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found!")
        print("   Please run ingest/clean_data.py first.")
        return None
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"✅ Loaded {len(documents)} documents")
    return documents

# ============================================
# STEP 2: INITIALIZE EMBEDDING MODEL
# ============================================

def load_embedding_model():
    """Load sentence-transformers model"""
    print(f"\n🤖 Loading embedding model: {EMBEDDING_MODEL}...")
    print("   (This may take a moment on first run - model will be downloaded)")
    
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Model loaded successfully!")
        print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# ============================================
# STEP 3: CREATE EMBEDDINGS
# ============================================

def create_embeddings(documents, model):
    """
    Create embeddings for all documents
    Returns: numpy array of embeddings
    """
    print("\n🔢 Creating embeddings...")
    print(f"   Processing {len(documents)} documents...")
    
    # Extract text from documents
    texts = [doc['text'] for doc in documents]
    
    # Create embeddings with progress bar
    print("\n   Generating embeddings (this may take a few minutes)...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,  # Adjust based on your RAM
        convert_to_numpy=True
    )
    
    print(f"\n✅ Created embeddings!")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dimension: {embeddings.shape[1]}")
    
    return embeddings

# ============================================
# STEP 4: SAVE EMBEDDINGS
# ============================================

def save_embeddings(embeddings, documents):
    """Save embeddings and documents"""
    print("\n💾 Saving embeddings...")
    
    # Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Save embeddings as pickle (numpy array)
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"✅ Embeddings saved to: {EMBEDDINGS_FILE}")
    print(f"   File size: {os.path.getsize(EMBEDDINGS_FILE) / (1024*1024):.2f} MB")
    
    # Add embedding info to documents and save
    for i, doc in enumerate(documents):
        doc['embedding_index'] = i
        doc['embedding_dim'] = embeddings.shape[1]
    
    with open(DOCUMENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Documents with metadata saved to: {DOCUMENTS_FILE}")
    
    return EMBEDDINGS_FILE, DOCUMENTS_FILE

# ============================================
# STEP 5: VERIFY EMBEDDINGS
# ============================================

def verify_embeddings(embeddings, documents):
    """Verify embeddings are created correctly"""
    print("\n🔍 Verifying embeddings...")
    
    # Check dimensions
    assert len(embeddings) == len(documents), "Mismatch in number of embeddings and documents!"
    print(f"✅ Count check passed: {len(embeddings)} embeddings for {len(documents)} documents")
    
    # Check for NaN values
    nan_count = np.isnan(embeddings).sum()
    if nan_count > 0:
        print(f"⚠️  Warning: Found {nan_count} NaN values in embeddings")
    else:
        print(f"✅ No NaN values found")
    
    # Show sample embedding
    print(f"\n📊 Sample embedding (first 10 dimensions):")
    print(f"   Document: {documents[0]['id']}")
    print(f"   Embedding: {embeddings[0][:10]}")
    
    # Calculate embedding stats
    print(f"\n📈 Embedding Statistics:")
    print(f"   Mean: {np.mean(embeddings):.6f}")
    print(f"   Std: {np.std(embeddings):.6f}")
    print(f"   Min: {np.min(embeddings):.6f}")
    print(f"   Max: {np.max(embeddings):.6f}")

# ============================================
# MAIN PIPELINE
# ============================================

def main():
    """Main embedding creation pipeline"""
    
    print("="*60)
    print("  EMBEDDING CREATION PIPELINE")
    print("="*60)
    
    # Step 1: Load documents
    print("\n" + "="*60)
    print("STEP 1: LOAD CLEANED DOCUMENTS")
    print("="*60)
    documents = load_cleaned_documents()
    if documents is None:
        return
    
    # Step 2: Load model
    print("\n" + "="*60)
    print("STEP 2: LOAD EMBEDDING MODEL")
    print("="*60)
    model = load_embedding_model()
    if model is None:
        return
    
    # Step 3: Create embeddings
    print("\n" + "="*60)
    print("STEP 3: CREATE EMBEDDINGS")
    print("="*60)
    embeddings = create_embeddings(documents, model)
    
    # Step 4: Save embeddings
    print("\n" + "="*60)
    print("STEP 4: SAVE EMBEDDINGS")
    print("="*60)
    embeddings_file, docs_file = save_embeddings(embeddings, documents)
    
    # Step 5: Verify
    print("\n" + "="*60)
    print("STEP 5: VERIFY EMBEDDINGS")
    print("="*60)
    verify_embeddings(embeddings, documents)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ EMBEDDING CREATION COMPLETE!")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   - Documents processed: {len(documents)}")
    print(f"   - Embeddings created: {len(embeddings)}")
    print(f"   - Embedding dimension: {embeddings.shape[1]}")
    print(f"   - Model used: {EMBEDDING_MODEL}")
    print(f"\n📁 Output files:")
    print(f"   - Embeddings: {embeddings_file}")
    print(f"   - Documents: {docs_file}")
    print(f"\n🎯 Ready for vector store creation!")

if __name__ == "__main__":
    main()