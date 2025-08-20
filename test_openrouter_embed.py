import os, requests, json
from dotenv import load_dotenv
load_dotenv()

AI_API_KEY = os.getenv("AI_API_KEY")
EMB_MODEL = "openai/text-embedding-3-small"
url = "https://openrouter.ai/api/v1/embeddings"
headers = {
    "Authorization": f"Bearer {AI_API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:5000",
    "X-Title": "Universal Doc Analyzer",
    "User-Agent": "Universal-Doc-Analyzer/1.0",
}
payload = {"model": EMB_MODEL, "input": ["hello world"]}

r = requests.post(url, json=payload, headers=headers, timeout=60, allow_redirects=False)
print("STATUS:", r.status_code)
print("HEADERS:", r.headers)
print("BODY:", r.text[:800])
