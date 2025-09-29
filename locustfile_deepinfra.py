from locust import HttpUser, task, between
import os, json, random
from datetime import datetime
from dotenv import load_dotenv

# Load .env
load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")

# Config
REQUESTS_DIR = "sample_requests"
FORCED_MODEL = "google/gemma-3-12b-it"
LOG_FILE = "locust_responses.log"

def log_entry(request_name: str, status_code: int, user_text: str, llm_text: str, usage: dict | None):
    """Append simplified request/response details to log file."""
    ts = datetime.utcnow().isoformat()

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"[{ts}] {request_name} [{status_code}]\n")
        f.write("-" * 80 + "\n")

        f.write("USER REQUEST (truncated):\n")
        f.write(user_text[:500] + "\n\n")

        f.write("LLM RESPONSE (truncated):\n")
        f.write(llm_text[:1000] + "\n\n")

        if usage:
            f.write("USAGE:\n")
            f.write(f"  prompt_tokens: {usage.get('prompt_tokens')}\n")
            f.write(f"  completion_tokens: {usage.get('completion_tokens')}\n")
            f.write(f"  total_tokens: {usage.get('total_tokens')}\n")
            if "estimated_cost" in usage:
                f.write(f"  estimated_cost: {usage.get('estimated_cost')}\n")

        f.write("=" * 80 + "\n\n")

def load_requests(folder: str = REQUESTS_DIR):
    requests_data = []
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Always override model
                data["model"] = FORCED_MODEL
                requests_data.append({"filename": fname, "content": data})
    return requests_data

class DeepInfraUser(HttpUser):
    wait_time = between(1, 3)
    requests_data = load_requests()

    @task
    def run_random_request(self):
        if not self.requests_data:
            return

        req = random.choice(self.requests_data)
        payload = req["content"]
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        with self.client.post(
            "/v1/openai/chat/completions",
            json=payload,
            headers=headers,
            name=f"preconfigured_{req['filename']}",
            catch_response=True
        ) as response:
            user_text = json.dumps(payload, ensure_ascii=False)[:500]
            llm_text, usage = "", None

            try:
                resp_json = response.json()
                if "choices" in resp_json and resp_json["choices"]:
                    llm_text = resp_json["choices"][0]["message"].get("content", "")
                usage = resp_json.get("usage")
            except Exception as e:
                llm_text = f"[Error parsing response: {e}]"

            log_entry(req["filename"], response.status_code, user_text, llm_text, usage)

            if response.status_code != 200:
                response.failure(f"Bad status: {response.status_code}")
