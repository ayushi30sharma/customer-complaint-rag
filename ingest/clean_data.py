import pandas as pd
import os
import json
import re

# ============================================
# CONFIGURATION
# ============================================

DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

COMPLAINTS_FILE = os.path.join(DATA_DIR, "ecommerce_customer_complaint_records.csv")
RESOLUTIONS_FILE = os.path.join(DATA_DIR, "ecommerce_customer_support_resolution_notes.csv")
RELEASES_FILE = os.path.join(DATA_DIR, "ecommerce_product_release_notes.csv")

# ============================================
# STEP 1: TEXT CLEANING FUNCTIONS
# ============================================

def clean_text(text):
    """
    Clean and normalize text data
    - Remove extra whitespace
    - Remove special characters
    - Convert to lowercase
    - Handle missing values
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
    text = re.sub(r'[^a-z0-9\s\.\,\!\?\-]', '', text)
    
    # Remove extra spaces again after special char removal
    text = ' '.join(text.split())
    
    # Strip leading/trailing spaces
    text = text.strip()
    
    return text

def clean_complaints(complaints_df):
    """Clean complaint text columns"""
    print("\n🧹 Cleaning complaint text...")
    
    # Identify text columns to clean (adjust based on your CSV)
    text_columns = complaints_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in text_columns:
        complaints_df[f'{col}_cleaned'] = complaints_df[col].apply(clean_text)
    
    print(f"✅ Cleaned {len(text_columns)} complaint columns")
    return complaints_df

def clean_resolutions(resolutions_df):
    """Clean resolution note text columns"""
    print("\n🧹 Cleaning resolution text...")
    
    text_columns = resolutions_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in text_columns:
        resolutions_df[f'{col}_cleaned'] = resolutions_df[col].apply(clean_text)
    
    print(f"✅ Cleaned {len(text_columns)} resolution columns")
    return resolutions_df

def clean_releases(releases_df):
    """Clean product release note text columns"""
    print("\n🧹 Cleaning release note text...")
    
    text_columns = releases_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in text_columns:
        releases_df[f'{col}_cleaned'] = releases_df[col].apply(clean_text)
    
    print(f"✅ Cleaned {len(text_columns)} release columns")
    return releases_df

# ============================================
# STEP 2: MERGE COMPLAINTS WITH RESOLUTIONS
# ============================================

def merge_complaints_resolutions(complaints_df, resolutions_df):
    """
    Merge complaints with their resolution notes
    Assumes both have a common key (e.g., complaint_id, ticket_id)
    Adjust the merge key based on your actual CSV columns
    """
    print("\n🔗 Merging complaints with resolutions...")
    
    # Display columns to help identify merge key
    print(f"Complaint columns: {complaints_df.columns.tolist()}")
    print(f"Resolution columns: {resolutions_df.columns.tolist()}")
    
    # Try to identify common key columns
    # Common patterns: id, complaint_id, ticket_id, case_id, etc.
    possible_keys = ['id', 'complaint_id', 'ticket_id', 'case_id', 'record_id']
    
    merge_key = None
    for key in possible_keys:
        if key in complaints_df.columns and key in resolutions_df.columns:
            merge_key = key
            break
    
    if merge_key:
        # Perform merge
        merged_df = complaints_df.merge(
            resolutions_df, 
            on=merge_key, 
            how='left',  # Keep all complaints even if no resolution
            suffixes=('_complaint', '_resolution')
        )
        print(f"✅ Merged on key: '{merge_key}'")
        print(f"   Total merged records: {len(merged_df)}")
    else:
        # If no common key, just concatenate by index
        print("⚠️  No common merge key found. Using index-based merge.")
        merged_df = pd.concat([complaints_df, resolutions_df], axis=1)
    
    return merged_df

# ============================================
# STEP 3: ATTACH PRODUCT RELEASE CONTEXT
# ============================================

def attach_release_context(merged_df, releases_df):
    """
    Attach relevant product release notes to complaints
    This can be based on:
    - Product ID
    - Date range (complaints after a release)
    - Category/Product name
    """
    print("\n📎 Attaching product release context...")
    
    # Convert releases to a context string
    release_context = "\n".join([
        f"Release: {row.to_dict()}" 
        for _, row in releases_df.iterrows()
    ])
    
    # Add release context to each merged record
    merged_df['release_context'] = release_context
    
    print(f"✅ Attached release context to {len(merged_df)} records")
    return merged_df

# ============================================
# STEP 4: CREATE CLEAN RAG DOCUMENTS
# ============================================

def create_rag_documents(merged_df):
    """
    Create final clean documents for RAG
    Each document contains:
    - id: unique identifier
    - text: combined cleaned text from all sources
    - source: type of document
    - metadata: original data for reference
    """
    print("\n📄 Creating RAG documents...")
    
    documents = []
    
    for idx, row in merged_df.iterrows():
        # Combine all cleaned text columns
        cleaned_columns = [col for col in row.index if '_cleaned' in col]
        combined_text = ' | '.join([
            f"{col.replace('_cleaned', '')}: {row[col]}" 
            for col in cleaned_columns 
            if pd.notna(row[col]) and row[col] != ""
        ])
        
        # Create document
        doc = {
            'id': f"doc_{idx}",
            'text': combined_text,
            'source': 'merged_complaint_resolution',
            'metadata': {
                'original_index': idx,
                'has_resolution': 'resolution' in str(row.to_dict()),
                'record': row.to_dict()
            }
        }
        
        documents.append(doc)
    
    print(f"✅ Created {len(documents)} RAG documents")
    return documents

# ============================================
# STEP 5: SAVE OUTPUT
# ============================================

def save_cleaned_documents(documents):
    """Save cleaned documents to JSON"""
    print("\n💾 Saving cleaned documents...")
    
    # Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Save as JSON
    output_file = os.path.join(PROCESSED_DIR, "cleaned_rag_documents.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved to: {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    
    # Display sample
    print("\n" + "="*60)
    print("SAMPLE DOCUMENT")
    print("="*60)
    sample = json.dumps(documents[0], indent=2)
    print(sample[:800] + "\n..." if len(sample) > 800 else sample)
    
    return output_file

# ============================================
# MAIN PIPELINE
# ============================================

def main():
    """Main data cleaning pipeline"""
    
    print("="*60)
    print("  DATA CLEANING PIPELINE FOR RAG")
    print("="*60)
    
    # Load datasets
    print("\n📥 Loading datasets...")
    try:
        complaints = pd.read_csv(COMPLAINTS_FILE)
        resolutions = pd.read_csv(RESOLUTIONS_FILE)
        releases = pd.read_csv(RELEASES_FILE)
        print(f"✅ Loaded:")
        print(f"   - {len(complaints)} complaints")
        print(f"   - {len(resolutions)} resolutions")
        print(f"   - {len(releases)} release notes")
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        return
    
    # Step 1: Clean text
    print("\n" + "="*60)
    print("STEP 1: CLEAN TEXT")
    print("="*60)
    complaints = clean_complaints(complaints)
    resolutions = clean_resolutions(resolutions)
    releases = clean_releases(releases)
    
    # Step 2: Merge complaints with resolutions
    print("\n" + "="*60)
    print("STEP 2: MERGE COMPLAINTS + RESOLUTIONS")
    print("="*60)
    merged = merge_complaints_resolutions(complaints, resolutions)
    
    # Step 3: Attach release context
    print("\n" + "="*60)
    print("STEP 3: ATTACH RELEASE CONTEXT")
    print("="*60)
    merged = attach_release_context(merged, releases)
    
    # Step 4: Create RAG documents
    print("\n" + "="*60)
    print("STEP 4: CREATE RAG DOCUMENTS")
    print("="*60)
    documents = create_rag_documents(merged)
    
    # Step 5: Save output
    print("\n" + "="*60)
    print("STEP 5: SAVE OUTPUT")
    print("="*60)
    output_file = save_cleaned_documents(documents)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ DATA CLEANING COMPLETE!")
    print("="*60)
    print(f"📊 Statistics:")
    print(f"   - Input complaints: {len(complaints)}")
    print(f"   - Input resolutions: {len(resolutions)}")
    print(f"   - Input releases: {len(releases)}")
    print(f"   - Output RAG documents: {len(documents)}")
    print(f"\n📁 Output file: {output_file}")
    print("\n🎯 Ready for embedding generation!")

if __name__ == "__main__":
    main()