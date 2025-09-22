from locust import HttpUser, task, between
import os, base64
from datetime import datetime
from dotenv import load_dotenv

# Load .env
load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")

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

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def make_prompt(n_tokens: int) -> str:
    return " ".join(["token"] * n_tokens) + "Ignore the list of tokens previously provided. Now tell me the first word that appears in each of the two images. Then list and describe 20 different fruits, each with a short description."

class DeepInfraUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def chat_completion_with_images(self):
        # Fixed token sizes
        input_tokens = 10000
        output_tokens = 1000
        user_prompt = make_prompt(input_tokens)

        # Encode your two images
        img1_b64 = image_to_base64("images/full_page_lorem_large.png")
        img2_b64 = image_to_base64("images/subset_graph.png")

        # Build multimodal messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img1_b64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img2_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]

        payload = {
            # Change model if needed to one of DeepInfra's multimodal ones
            "model": "meta-llama/Llama-3.2-90B-Vision-Instruct",
            "messages": messages,
            "max_tokens": output_tokens,
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        with self.client.post(
            "/v1/openai/chat/completions",
            json=payload,
            headers=headers,
            name="chat_completion_with_images",
            catch_response=True
        ) as response:
            user_text = user_prompt
            llm_text, usage = "", None

            try:
                resp_json = response.json()
                if "choices" in resp_json and resp_json["choices"]:
                    llm_text = resp_json["choices"][0]["message"].get("content", "")
                usage = resp_json.get("usage")
            except Exception as e:
                llm_text = f"[Error parsing response: {e}]"

            log_entry("chat_completion_with_images", response.status_code, user_text, llm_text, usage)

            if response.status_code != 200:
                response.failure(f"Bad status: {response.status_code}")
