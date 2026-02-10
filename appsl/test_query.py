import requests
import json

BASE_URL = "http://localhost:8000"

# Test query endpoint
print("="*60)
print("TESTING QUERY ENDPOINT")
print("="*60)

query_data = {
    "query": "How do you handle damaged products?",
    "top_k": 3,
    "max_tokens": 200,
    "temperature": 0.7
}

response = requests.post(f"{BASE_URL}/query", json=query_data)

print(f"Status Code: {response.status_code}")
print("\nResponse:")
print(json.dumps(response.json(), indent=2))