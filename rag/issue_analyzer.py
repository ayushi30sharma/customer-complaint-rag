import json
import os
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from retriever import ComplaintRetriever

# ============================================
# CONFIGURATION
# ============================================

PROCESSED_DIR = os.path.join("data", "processed")
DOCUMENTS_FILE = os.path.join(PROCESSED_DIR, "vector_store", "documents.json")

# ============================================
# ISSUE ANALYZER CLASS
# ============================================

class RecurringIssueAnalyzer:
    """
    Analyze customer complaints to identify recurring issues
    """
    
    def __init__(self):
        """Initialize analyzer"""
        self.documents = None
        self.embeddings = None
        self.embedding_model = None
        self.clusters = None
        
    def load_data(self):
        """Load documents and initialize embedding model"""
        print("📥 Loading data for issue analysis...")
        
        # Load documents
        if not os.path.exists(DOCUMENTS_FILE):
            raise FileNotFoundError(f"Documents not found: {DOCUMENTS_FILE}")
        
        with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        print(f"✅ Loaded {len(self.documents)} documents")
        
        # Load embedding model
        print("🤖 Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")
        
        return self
    
    def create_embeddings(self):
        """Create embeddings for all documents"""
        print("\n🔢 Creating embeddings for clustering...")
        
        texts = [doc['text'] for doc in self.documents]
        self.embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✅ Created embeddings: {self.embeddings.shape}")
        return self
    
    def cluster_complaints(self, n_clusters=5):
        """
        Cluster complaints to identify recurring issues
        
        Args:
            n_clusters (int): Number of issue categories to identify
        
        Returns:
            dict: Clustering results
        """
        print(f"\n🔍 Clustering complaints into {n_clusters} issue categories...")
        
        if self.embeddings is None:
            raise RuntimeError("Embeddings not created. Call create_embeddings() first.")
        
        # Filter only complaints (not resolutions or releases)
        complaint_indices = [
            i for i, doc in enumerate(self.documents) 
            if doc['source'] == 'complaint'
        ]
        
        if not complaint_indices:
            print("⚠️  No complaints found!")
            return None
        
        complaint_embeddings = self.embeddings[complaint_indices]
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(complaint_embeddings)
        
        # Organize results
        clusters = defaultdict(list)
        for idx, label in zip(complaint_indices, cluster_labels):
            clusters[label].append({
                'doc_id': self.documents[idx]['id'],
                'text': self.documents[idx]['text'],
                'metadata': self.documents[idx].get('metadata', {})
            })
        
        self.clusters = clusters
        
        print(f"✅ Clustering complete!")
        print(f"\nCluster distribution:")
        for cluster_id, docs in clusters.items():
            print(f"  Cluster {cluster_id}: {len(docs)} complaints")
        
        return clusters
    
    def analyze_cluster_themes(self, top_words=10):
        """
        Analyze each cluster to identify common themes
        
        Args:
            top_words (int): Number of top keywords to show per cluster
        
        Returns:
            dict: Theme analysis for each cluster
        """
        if self.clusters is None:
            raise RuntimeError("Clusters not created. Call cluster_complaints() first.")
        
        print(f"\n📊 Analyzing themes for each cluster...")
        
        from collections import Counter
        import re
        
        # Stop words to exclude
        stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your'
        ])
        
        cluster_themes = {}
        
        for cluster_id, docs in self.clusters.items():
            # Extract all words from cluster documents
            all_text = ' '.join([doc['text'].lower() for doc in docs])
            words = re.findall(r'\b[a-z]{3,}\b', all_text)  # Words 3+ chars
            
            # Filter stop words
            filtered_words = [w for w in words if w not in stop_words]
            
            # Count word frequencies
            word_counts = Counter(filtered_words)
            top_keywords = word_counts.most_common(top_words)
            
            # Sample complaints
            sample_docs = docs[:3]  # First 3 complaints
            
            cluster_themes[cluster_id] = {
                'size': len(docs),
                'top_keywords': top_keywords,
                'sample_complaints': [
                    {
                        'id': doc['doc_id'],
                        'preview': doc['text'][:150] + '...'
                    }
                    for doc in sample_docs
                ],
                'percentage': (len(docs) / sum(len(d) for d in self.clusters.values())) * 100
            }
        
        return cluster_themes
    
    def compare_before_after_release(self, release_keyword='release', date_field='date'):
        """
        Compare complaints before vs after product releases
        
        Args:
            release_keyword (str): Keyword to identify releases
            date_field (str): Field name for dates in metadata
        
        Returns:
            dict: Comparison analysis
        """
        print(f"\n📅 Comparing complaints before/after product releases...")
        
        # Identify release documents
        releases = [
            doc for doc in self.documents 
            if doc['source'] == 'release'
        ]
        
        # Identify complaints
        complaints = [
            doc for doc in self.documents 
            if doc['source'] == 'complaint'
        ]
        
        if not releases:
            print("⚠️  No release notes found!")
            return None
        
        print(f"✅ Found {len(releases)} releases and {len(complaints)} complaints")
        
        # Simple analysis: count complaints by source
        complaint_sources = Counter([c.get('metadata', {}).get('source', 'unknown') for c in complaints])
        
        analysis = {
            'total_releases': len(releases),
            'total_complaints': len(complaints),
            'complaint_distribution': dict(complaint_sources),
            'release_topics': [r['text'][:100] + '...' for r in releases[:5]]
        }
        
        return analysis
    
    def generate_insights(self, cluster_themes):
        """
        Generate actionable insights from analysis
        
        Args:
            cluster_themes (dict): Theme analysis results
        
        Returns:
            dict: Insights and recommendations
        """
        print(f"\n💡 Generating insights...")
        
        insights = {
            'top_issues': [],
            'recommendations': [],
            'priority_areas': []
        }
        
        # Sort clusters by size
        sorted_clusters = sorted(
            cluster_themes.items(),
            key=lambda x: x[1]['size'],
            reverse=True
        )
        
        for cluster_id, theme in sorted_clusters[:3]:  # Top 3 issues
            issue = {
                'cluster_id': cluster_id,
                'complaint_count': theme['size'],
                'percentage': theme['percentage'],
                'keywords': [kw[0] for kw in theme['top_keywords'][:5]],
                'severity': 'High' if theme['percentage'] > 30 else 'Medium' if theme['percentage'] > 15 else 'Low'
            }
            insights['top_issues'].append(issue)
        
        # Generate recommendations
        for issue in insights['top_issues']:
            if 'damaged' in issue['keywords'] or 'shipping' in issue['keywords']:
                insights['recommendations'].append(
                    "Improve packaging and shipping quality control"
                )
            elif 'delay' in issue['keywords'] or 'late' in issue['keywords']:
                insights['recommendations'].append(
                    "Optimize logistics and delivery timelines"
                )
            elif 'refund' in issue['keywords'] or 'return' in issue['keywords']:
                insights['recommendations'].append(
                    "Streamline refund/return process"
                )
            elif 'quality' in issue['keywords'] or 'defect' in issue['keywords']:
                insights['recommendations'].append(
                    "Enhance product quality testing"
                )
        
        # Remove duplicates
        insights['recommendations'] = list(set(insights['recommendations']))
        
        return insights
    
    def display_analysis_report(self, cluster_themes, insights):
        """Display formatted analysis report"""
        print("\n" + "="*70)
        print("  RECURRING ISSUES ANALYSIS REPORT")
        print("="*70)
        
        # Top Issues
        print("\n📌 TOP RECURRING ISSUES:")
        print("-"*70)
        for i, issue in enumerate(insights['top_issues'], 1):
            print(f"\n{i}. Issue Cluster {issue['cluster_id']}")
            print(f"   Complaints: {issue['complaint_count']} ({issue['percentage']:.1f}%)")
            print(f"   Severity: {issue['severity']}")
            print(f"   Keywords: {', '.join(issue['keywords'])}")
        
        # Detailed Themes
        print("\n" + "="*70)
        print("📊 DETAILED CLUSTER ANALYSIS:")
        print("="*70)
        
        for cluster_id, theme in cluster_themes.items():
            print(f"\n--- Cluster {cluster_id} ---")
            print(f"Size: {theme['size']} complaints ({theme['percentage']:.1f}%)")
            print(f"\nTop Keywords:")
            for word, count in theme['top_keywords'][:5]:
                print(f"  - {word}: {count}")
            print(f"\nSample Complaints:")
            for i, sample in enumerate(theme['sample_complaints'], 1):
                print(f"  {i}. {sample['preview']}")
        
        # Recommendations
        print("\n" + "="*70)
        print("💡 RECOMMENDATIONS:")
        print("="*70)
        for i, rec in enumerate(insights['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*70)

# ============================================
# TEST ANALYZER
# ============================================

def test_analyzer():
    """Test the recurring issue analyzer"""
    print("="*70)
    print("  TESTING RECURRING ISSUE ANALYZER")
    print("="*70)
    
    # Initialize
    analyzer = RecurringIssueAnalyzer()
    analyzer.load_data()
    
    # Create embeddings
    analyzer.create_embeddings()
    
    # Cluster complaints
    n_clusters = 5  # Adjust based on your data size
    clusters = analyzer.cluster_complaints(n_clusters=n_clusters)
    
    # Analyze themes
    cluster_themes = analyzer.analyze_cluster_themes(top_words=10)
    
    # Compare before/after releases
    release_comparison = analyzer.compare_before_after_release()
    
    if release_comparison:
        print("\n" + "="*70)
        print("📅 RELEASE COMPARISON:")
        print("="*70)
        print(f"Total Releases: {release_comparison['total_releases']}")
        print(f"Total Complaints: {release_comparison['total_complaints']}")
    
    # Generate insights
    insights = analyzer.generate_insights(cluster_themes)
    
    # Display report
    analyzer.display_analysis_report(cluster_themes, insights)
    
    print("\n✅ Analysis complete!")

# ============================================
# MAIN
# ============================================

def main():
    """Main function"""
    test_analyzer()

if __name__ == "__main__":
    main()