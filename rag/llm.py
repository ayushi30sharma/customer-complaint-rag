import requests
import json
from typing import Optional, List, Dict

# ============================================
# CONFIGURATION
# ============================================

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:latest"  # Change to "mistral" or "llama3.1" if you prefer

# ============================================
# LLM CLASS
# ============================================

class OllamaLLM:
    """
    Wrapper for Ollama LLM (LLaMA/Mistral)
    """
    
    def __init__(self, model_name=DEFAULT_MODEL, base_url=OLLAMA_BASE_URL):
        """
        Initialize Ollama LLM
        
        Args:
            model_name (str): Name of the Ollama model
            base_url (str): Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is running and model is available"""
        print(f"🔌 Connecting to Ollama at {self.base_url}...")
        
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if self.model_name not in model_names:
                print(f"⚠️  Model '{self.model_name}' not found!")
                print(f"   Available models: {model_names}")
                print(f"\n   To install the model, run:")
                print(f"   ollama pull {self.model_name}")
                raise ValueError(f"Model {self.model_name} not available")
            
            print(f"✅ Connected to Ollama")
            print(f"✅ Model '{self.model_name}' is ready")
            
        except requests.exceptions.ConnectionError:
            print(f"❌ Error: Cannot connect to Ollama at {self.base_url}")
            print(f"\n   Please make sure:")
            print(f"   1. Ollama is installed (https://ollama.com)")
            print(f"   2. Ollama service is running")
            print(f"   3. Run 'ollama serve' in terminal")
            raise
    
    def generate(self, prompt, max_tokens=500, temperature=0.7, stream=False):
        """
        Generate response from LLM
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature (0.0 - 1.0)
            stream (bool): Whether to stream response
        
        Returns:
            str: Generated text
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            full_response += chunk['response']
                            print(chunk['response'], end='', flush=True)
                print()  # New line after streaming
                return full_response
            else:
                # Handle non-streaming response
                result = response.json()
                return result['response']
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error generating response: {e}")
            return None
    
    def generate_with_context(self, query, context, max_tokens=500, temperature=0.7):
        """
        Generate response using retrieved context (for RAG)
        
        Args:
            query (str): User query
            context (str): Retrieved context from vector store
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
        
        Returns:
            str: Generated answer
        """
        prompt = f"""You are a helpful customer support assistant. Use the following context to answer the user's question accurately and concisely.

Context:
{context}

User Question: {query}

Answer: Provide a clear and helpful response based on the context above. If the context doesn't contain relevant information, say so politely."""

        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

# ============================================
# TEST LLM
# ============================================

def test_llm():
    """Test the LLM connection and generation"""
    print("="*60)
    print("  TESTING OLLAMA LLM")
    print("="*60)
    
    # Initialize LLM
    try:
        llm = OllamaLLM()
    except Exception as e:
        print(f"\n❌ Failed to initialize LLM: {e}")
        return
    
    # Test 1: Simple generation
    print("\n" + "="*60)
    print("TEST 1: SIMPLE GENERATION")
    print("="*60)
    
    test_prompt = "What is customer service? Answer in 2 sentences."
    print(f"\nPrompt: {test_prompt}")
    print("\nResponse:")
    print("-" * 60)
    
    response = llm.generate(test_prompt, max_tokens=100, temperature=0.7)
    print(response)
    
    # Test 2: RAG-style generation with context
    print("\n" + "="*60)
    print("TEST 2: RAG-STYLE GENERATION")
    print("="*60)
    
    sample_context = """
[Document 1]
Source: complaint
Text: Customer complained about damaged product during shipping. Resolution: Full refund issued and replacement sent with expedited shipping.

[Document 2]
Source: resolution
Text: Product quality issue resolved by offering 20% discount on next purchase and apology letter sent.
"""
    
    test_query = "How do you handle damaged products?"
    print(f"\nQuery: {test_query}")
    print(f"\nContext: {sample_context[:150]}...")
    print("\nResponse:")
    print("-" * 60)
    
    response = llm.generate_with_context(
        query=test_query,
        context=sample_context,
        max_tokens=200,
        temperature=0.7
    )
    print(response)
    
    print("\n✅ LLM test complete!")

# ============================================
# MAIN
# ============================================

def main():
    """Main function"""
    test_llm()

if __name__ == "__main__":
    main()