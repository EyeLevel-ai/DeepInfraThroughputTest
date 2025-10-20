"""
LLM Load Tester → CSV Logger (DeepInfra Gemma 3-12B)
Continuously ramps up concurrency and logs each request as a CSV row:
  - timestamp_start
  - timestamp_end
  - status_code
  - input_tokens
  - output_tokens
  - time_to_first_token
  - concurrency_at_start
  - source_filename
  - truncated_output
"""

import os, json, asyncio, time, random, signal, csv
from datetime import datetime
from dotenv import load_dotenv
import httpx
from transformers import AutoTokenizer
from typing import Dict

# ==========================================
#  CONFIG
# ==========================================
load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")
if not API_KEY:
    raise ValueError("DEEPINFRA_API_KEY not found in .env")

REQUESTS_DIR = "sample_requests"
FORCED_MODEL = "google/gemma-3-12b-it"
ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"
STREAM = False  # DeepInfra doesn’t use streaming like Kvant

CSV_FILE = "llm_load_results_deepinfra.csv"

# --- Load scaling ---
INITIAL_CONCURRENCY = 2
RAMP_RATE = 1
RAMP_INTERVAL = 2.0
MAX_CONCURRENCY = 500
RAMP_COOLDOWN = 0.1

# ==========================================
#  Tokenizer
# ==========================================
print("🔤 Loading Gemma 3-12B tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-12b-it",
    token=os.getenv("HF_TOKEN")
)
print("✅ Tokenizer loaded.")


# ==========================================
#  Request utilities
# ==========================================
def extract_user_text(data: dict) -> str:
    """Extract user text from messages for token counting."""
    text = ""
    for m in data.get("messages", []):
        if m["role"] == "user":
            if isinstance(m["content"], list):
                for c in m["content"]:
                    if isinstance(c, dict) and "text" in c:
                        text += c["text"]
            elif isinstance(m["content"], str):
                text += m["content"]
    return text


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0


def load_requests(folder: str = REQUESTS_DIR):
    """Load and preprocess request JSON files."""
    reqs = []
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["model"] = FORCED_MODEL
            user_text = extract_user_text(data)
            input_tokens = count_tokens(user_text)
            reqs.append({
                "filename": fname,
                "content": data,
                "input_tokens": input_tokens
            })
    if not reqs:
        raise RuntimeError(f"No .json files found in {folder}")
    print(f"✅ Loaded {len(reqs)} request templates.")
    return reqs


# ==========================================
#  CSV writer
# ==========================================
def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_start",
                "timestamp_end",
                "status_code",
                "input_tokens",
                "output_tokens",
                "time_to_first_token",
                "concurrency_at_start",
                "source_filename",
                "truncated_output"
            ])


def append_csv_row(row: list):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ==========================================
#  LLM Request Logic
# ==========================================
class LLMUser:
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key

    async def send_request(self, req: Dict, concurrency: int):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start_time = datetime.utcnow()
        start_perf = time.perf_counter()
        first_token_time = None
        output_text = ""
        status_code = None

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.post(self.endpoint, headers=headers, json=req["content"])
                status_code = resp.status_code

                if status_code == 200:
                    data = resp.json()
                    if "choices" in data and data["choices"]:
                        output_text = data["choices"][0]["message"].get("content", "")
                    if "usage" in data:
                        # optional: capture reported token counts
                        output_tokens = data["usage"].get("completion_tokens", 0)
                    else:
                        output_tokens = count_tokens(output_text)
                else:
                    output_tokens = 0
                    output_text = resp.text

        except Exception as e:
            status_code = -1
            output_tokens = 0
            output_text = f"Exception: {e}"

        first_token_time = time.perf_counter() - start_perf

        append_csv_row([
            start_time.isoformat(),
            datetime.utcnow().isoformat(),
            status_code,
            req["input_tokens"],
            output_tokens,
            round(first_token_time, 3),
            concurrency,
            req["filename"],
            output_text[:300].replace("\n", " ").replace("\r", " ")
        ])


# ==========================================
#  Stress Runner
# ==========================================
class LLMStressTest:
    def __init__(self, endpoint, api_key, requests_data):
        self.user = LLMUser(endpoint, api_key)
        self.requests_data = requests_data
        self.stop_flag = False
        self.active_concurrency = INITIAL_CONCURRENCY
        self.active_tasks = set()

    def stop(self):
        self.stop_flag = True

    async def ramp_up(self):
        while not self.stop_flag:
            if self.active_concurrency < MAX_CONCURRENCY:
                self.active_concurrency += RAMP_RATE
                print(f"[+] Increased concurrency → {self.active_concurrency}")
            await asyncio.sleep(RAMP_INTERVAL)

    async def dispatcher(self):
        while not self.stop_flag:
            while len(self.active_tasks) < self.active_concurrency:
                req = random.choice(self.requests_data)
                task = asyncio.create_task(
                    self.user.send_request(req, concurrency=len(self.active_tasks))
                )
                self.active_tasks.add(task)
                task.add_done_callback(lambda t: self.active_tasks.discard(t))
            await asyncio.sleep(RAMP_COOLDOWN)

    async def run(self):
        print(f"🚀 Starting DeepInfra load test (ramp {RAMP_RATE}/s → max {MAX_CONCURRENCY})")
        init_csv()

        ramp_task = asyncio.create_task(self.ramp_up())
        dispatcher_task = asyncio.create_task(self.dispatcher())

        try:
            while not self.stop_flag:
                await asyncio.sleep(1)
                print(f"↺ Active concurrency: {len(self.active_tasks)}")
        except asyncio.CancelledError:
            pass
        finally:
            self.stop_flag = True
            ramp_task.cancel()
            dispatcher_task.cancel()
            await asyncio.gather(ramp_task, dispatcher_task, return_exceptions=True)
            print("🛑 Stopped load test. CSV saved.")


# ==========================================
#  Main Entrypoint
# ==========================================
async def main():
    requests_data = load_requests()
    runner = LLMStressTest(ENDPOINT, API_KEY, requests_data)

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, runner.stop)
    loop.add_signal_handler(signal.SIGTERM, runner.stop)

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
