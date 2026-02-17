import requests
import json
from typing import Optional

# ============================================
# CONFIGURATION
# ============================================

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:latest"

# ============================================
# LLM CLASS
# ============================================

class OllamaLLM:
    """
    Wrapper for Ollama LLM (LLaMA/Mistral)
    Designed for analytical RAG-style responses
    """
    
    def __init__(self, model_name=DEFAULT_MODEL, base_url=OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self._verify_connection()
    
    # ----------------------------------------
    # Verify Ollama Connection
    # ----------------------------------------

    def _verify_connection(self):
        print(f"🔌 Connecting to Ollama at {self.base_url}...")
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if self.model_name not in model_names:
                print(f"⚠️  Model '{self.model_name}' not found!")
                print(f"Available models: {model_names}")
                print(f"\nRun this to install:")
                print(f"ollama pull {self.model_name}")
                raise ValueError(f"Model {self.model_name} not available")
            
            print(f"✅ Connected to Ollama")
            print(f"✅ Model '{self.model_name}' is ready")
            
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Ollama at {self.base_url}")
            print("Make sure Ollama is running using: ollama serve")
            raise

    # ----------------------------------------
    # Basic Generation
    # ----------------------------------------

    def generate(self, prompt, max_tokens=500, temperature=0.2, stream=False):
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
                timeout=90
            )
            response.raise_for_status()
            
            if stream:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            full_response += chunk['response']
                            print(chunk['response'], end='', flush=True)
                print()
                return full_response
            else:
                result = response.json()
                return result.get("response", "").strip()
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error generating response: {e}")
            return "Error generating response from LLM."

    # ----------------------------------------
    # RAG Analytical Generation
    # ----------------------------------------

    def generate_with_context(self, query, context, max_tokens=600, temperature=0.2):
        """
        Generate structured analytical response using retrieved complaint context
        """

        prompt = f"""
You are a senior product analyst analyzing structured customer complaint data.

Your responsibilities:
- Identify clear issue patterns.
- Infer realistic root causes based ONLY on the complaint data.
- Highlight affected platforms, versions, or features if mentioned.
- Provide actionable product or engineering recommendations.

STRICT RULES:
- Use ONLY the complaint data below.
- Do NOT suggest contacting support.
- Do NOT provide generic customer service responses.
- Do NOT hallucinate information.
- If data is limited, clearly state limitations but still summarize visible patterns.

Complaint Data:
{context}

User Question:
{query}

Respond in the following structured format:

1. Issue Summary:
2. Observed Patterns:
3. Possible Root Causes:
4. Recommended Actions:
5. Confidence Level (Low/Medium/High):
"""

        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)


# ============================================
# MAIN TEST
# ============================================

def main():
    print("="*60)
    print("TESTING UPDATED OLLAMA LLM")
    print("="*60)

    try:
        llm = OllamaLLM()
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return

    sample_context = """
Complaint 1:
User reports payment failure after entering card details on v2.0 (iOS).
---
Complaint 2:
Multiple users facing payment timeout issue on Web platform (v3.0).
---
Complaint 3:
Checkout error during payment authorization stage.
---
"""

    query = "Why are payments failing?"

    print("\nGenerated Response:\n")
    response = llm.generate_with_context(query, sample_context)
    print(response)


if __name__ == "__main__":
    main()
