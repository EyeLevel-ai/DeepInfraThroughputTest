import os
import requests
import json
from dotenv import load_dotenv

# Force load variables from .env file
load_dotenv()

API_KEY = os.getenv("DEEPINFRA_API_KEY")
if not API_KEY:
    raise RuntimeError("No DeepInfra API key found. Please set DEEPINFRA_API_KEY in your .env file.")

url = "https://api.deepinfra.com/v1/openai/chat/completions"

payload = {
    "model": "google/gemma-3-12b-it",  # replace with a model you have access to
    "messages": [
        {"role": "user", "content": "Hello, can you summarize yourself in one sentence?"}
    ],
    "max_tokens": 100
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

print("=== Request ===")
print(json.dumps(payload, indent=2))

response = requests.post(url, headers=headers, json=payload)

print("\n=== Response ===")
print(f"Status: {response.status_code}")
print(response.text)
