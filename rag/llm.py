from transformers import pipeline

# ============================================
# CONFIGURATION
# ============================================

DEFAULT_MODEL = "google/flan-t5-base"

# ============================================
# LLM CLASS
# ============================================

class HuggingFaceLLM:
    """
    Lightweight Hugging Face LLM wrapper
    Compatible with free CPU deployment (Hugging Face Spaces)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize Hugging Face LLM

        Args:
            model_name (str): Hugging Face model name
        """
        print(f"🔌 Loading Hugging Face model: {model_name}...")
        self.model_name = model_name

        self.llm = pipeline(
            task="text-generation",
            model=model_name,
            max_new_tokens=256
        )

        print("✅ Hugging Face LLM loaded successfully")

    def generate(self, prompt: str):
        """
        Generate response from LLM

        Args:
            prompt (str): Input prompt

        Returns:
            str: Generated response
        """
        result = self.llm(prompt)
        return result[0]["generated_text"]

    def generate_with_context(self, query: str, context: str):
        """
        Generate response using retrieved context (RAG)

        Args:
            query (str): User query
            context (str): Retrieved context

        Returns:
            str: Generated answer
        """
        prompt = f"""
You are a senior customer support analyst.

Use ONLY the information provided in the context below.
Identify patterns, root causes, and trends if applicable.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question:
{query}

Answer in a clear, professional, and structured manner.
"""

        return self.generate(prompt)


# ============================================
# TEST LLM
# ============================================

def test_llm():
    """Test the Hugging Face LLM"""
    print("=" * 60)
    print("  TESTING HUGGING FACE LLM")
    print("=" * 60)

    llm = HuggingFaceLLM()

    # Test 1: Simple generation
    print("\nTEST 1: SIMPLE GENERATION")
    prompt = "What is customer service? Answer in 2 sentences."
    print("\nResponse:")
    print(llm.generate(prompt))

    # Test 2: RAG-style generation
    print("\nTEST 2: RAG-STYLE GENERATION")

    sample_context = """
Customer complained about damaged product during shipping.
Resolution: Full refund issued and replacement sent.
Another complaint mentioned poor packaging quality.
"""

    query = "What is the root cause of these complaints?"

    print("\nResponse:")
    print(llm.generate_with_context(query, sample_context))

    print("\n✅ LLM test complete!")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    test_llm()