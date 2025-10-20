from locust import HttpUser, task, between
import os, json, random, time
from datetime import datetime
from dotenv import load_dotenv

# ==========================================
#  Load environment and config
# ==========================================
load_dotenv()
API_KEY = os.getenv("KVANT_4B_API_KEY")

if not API_KEY:
    raise ValueError("KVANT_4B_API_KEY not found in .env")

REQUESTS_DIR = "sample_requests"
FORCED_MODEL = "inference-gemma-12b-it"
LOG_FILE = "locust_kvant_4b_responses.log"


# ==========================================
#  Utility functions
# ==========================================
def log_entry(request_name: str, status_code: int, user_text: str, llm_text: str, usage: dict | None):
    """Append simplified request/response details to a log file."""
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


def restructure_messages(messages):
    """Ensure user/assistant alternation (merge consecutive same-role messages)."""
    replacement_messages = []
    for message in messages:
        if len(replacement_messages) == 0 or message["role"] != replacement_messages[-1]["role"]:
            replacement_messages.append(message)
        else:
            # merge content arrays
            if isinstance(message["content"], list) and isinstance(replacement_messages[-1]["content"], list):
                replacement_messages[-1]["content"].extend(message["content"])
            else:
                replacement_messages[-1]["content"] += message["content"]
    return replacement_messages


def load_requests(folder: str = REQUESTS_DIR):
    """Load and preprocess request JSON files."""
    requests_data = []
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # normalize message structure
            if "messages" in data:
                data["messages"] = restructure_messages(data["messages"])

            # override model
            data["model"] = FORCED_MODEL

            requests_data.append({"filename": fname, "content": data})
    return requests_data


# ==========================================
#  Locust User Definition
# ==========================================
class KvantUser(HttpUser):
    wait_time = between(1, 3)
    host = "https://summary-api-eyelevel.apps.eyelevel.kvant.cloud"
    requests_data = load_requests()

    @task
    def run_random_request(self):
        """Send a random preconfigured request to Kvant API."""
        if not self.requests_data:
            return

        req = random.choice(self.requests_data)
        payload = req["content"]
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        start = time.perf_counter()
        with self.client.post(
            "/chat/completions",
            json=payload,
            headers=headers,
            name=f"kvant_{req['filename']}",
            catch_response=True,
        ) as response:
            latency = round(time.perf_counter() - start, 3)
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
            else:
                response.success()
