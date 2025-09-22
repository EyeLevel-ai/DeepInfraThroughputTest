import os
import time
import base64
import csv
import itertools
from io import BytesIO
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ensure headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

# Load env
load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")

# Config
N_REQUESTS = 10
MAX_WORKERS = 50
OUTPUT_CSV = "grid_results.csv"
OUTPUT_PDF = "grid_report.pdf"

# Grid search dimensions
INPUT_TOKENS = [1000, 5000, 10000]
OUTPUT_TOKENS = [256, 512, 1000]

# Explicit image resolutions (fractions of 3400x4400)
IMAGE_RESOLUTIONS = {
    "850x1100": (850, 1100),     # 1/4
    "1700x2200": (1700, 2200),   # 1/2
    "2550x3300": (2550, 3300),   # 3/4
    "3400x4400": (3400, 4400),   # full
}
NUM_IMAGES = [0, 1, 2, 3]

# ---- THREAD-SAFE IMAGE GENERATOR ----
def generate_image_b64(width: int, height: int) -> str:
    """Generate a synthetic image with given pixel resolution and return base64 string."""
    fig = Figure(figsize=(width/100, height/100), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.imshow([[0.2, 0.5], [0.7, 0.9]], cmap="viridis")
    ax.set_title(f"{width}x{height} test image")
    ax.axis("off")

    buf = BytesIO()
    canvas.print_png(buf)  # safe for threads
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ---- PROMPT ----
def make_prompt(n_tokens: int) -> str:
    return " ".join(["token"] * n_tokens) + " Answer briefly."

# ---- PAYLOAD BUILDER ----
def make_payload(input_tokens, output_tokens, resolution_name, num_images):
    user_prompt = make_prompt(input_tokens)
    width, height = IMAGE_RESOLUTIONS[resolution_name]

    images = []
    for _ in range(num_images):
        img_b64 = generate_image_b64(width, height)
        images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })

    messages = [{
        "role": "user",
        "content": images + [{"type": "text", "text": user_prompt}]
    }]

    return {
        "model": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "messages": messages,
        "max_tokens": output_tokens,
    }

# ---- API CALL ----
def run_request(input_tokens, output_tokens, resolution_name, num_images):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = make_payload(input_tokens, output_tokens, resolution_name, num_images)

    start = time.perf_counter()
    try:
        r = requests.post(
            "https://api.deepinfra.com/v1/openai/chat/completions",
            json=payload,
            headers=headers,
            timeout=120,
        )
        latency = time.perf_counter() - start
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        cost = usage.get("estimated_cost", None)
        return latency, cost
    except Exception:
        latency = time.perf_counter() - start
        return latency, None

# ---- GRID LOOP (with tqdm) ----
def run_grid():
    combos = list(itertools.product(INPUT_TOKENS, OUTPUT_TOKENS, IMAGE_RESOLUTIONS.keys(), NUM_IMAGES))
    total_jobs = len(combos) * N_REQUESTS

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input_tokens", "output_tokens", "resolution", "num_images", "trial", "latency", "cost"])

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for (in_t, out_t, res_name, n_img) in combos:
                for trial in range(N_REQUESTS):
                    fut = executor.submit(run_request, in_t, out_t, res_name, n_img)
                    fut.meta = (in_t, out_t, res_name, n_img, trial + 1)
                    futures.append(fut)

            for future in tqdm(as_completed(futures), total=total_jobs, desc="Running grid search"):
                in_t, out_t, res_name, n_img, trial = future.meta
                latency, cost = future.result()
                writer.writerow([in_t, out_t, res_name, n_img, trial, latency, cost])

    print(f"Results written to {OUTPUT_CSV}")

# ---- PDF REPORT ----
def make_pdf():
    df = pd.read_csv(OUTPUT_CSV)

    agg = df.groupby(["input_tokens", "output_tokens", "resolution", "num_images"]).agg(
        avg_latency=("latency", "mean"),
        std_latency=("latency", "std"),
        avg_cost=("cost", "mean"),
        trials=("latency", "count")
    ).reset_index()

    with PdfPages(OUTPUT_PDF) as pdf:
        # Latency vs input tokens
        plt.figure(figsize=(8,6))
        for out_t in agg["output_tokens"].unique():
            subset = agg[agg["output_tokens"] == out_t]
            plt.plot(subset["input_tokens"], subset["avg_latency"], marker="o", label=f"out={out_t}")
        plt.xlabel("Input tokens")
        plt.ylabel("Avg latency (s)")
        plt.title("Latency vs Input Tokens")
        plt.legend()
        pdf.savefig(); plt.close()

        # Cost vs input tokens
        plt.figure(figsize=(8,6))
        for out_t in agg["output_tokens"].unique():
            subset = agg[agg["output_tokens"] == out_t]
            plt.plot(subset["input_tokens"], subset["avg_cost"], marker="o", label=f"out={out_t}")
        plt.xlabel("Input tokens")
        plt.ylabel("Avg cost")
        plt.title("Cost vs Input Tokens")
        plt.legend()
        pdf.savefig(); plt.close()

        # Latency heatmap
        pivot = agg.pivot_table(index="input_tokens", columns="output_tokens", values="avg_latency")
        plt.figure(figsize=(8,6))
        plt.imshow(pivot, cmap="viridis", aspect="auto")
        plt.colorbar(label="Avg latency (s)")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xlabel("Output tokens")
        plt.ylabel("Input tokens")
        plt.title("Latency Heatmap")
        pdf.savefig(); plt.close()

        # Resolution effect
        plt.figure(figsize=(8,6))
        for res in agg["resolution"].unique():
            subset = agg[agg["resolution"] == res]
            plt.scatter(subset["input_tokens"], subset["avg_latency"], label=res)
        plt.xlabel("Input tokens")
        plt.ylabel("Avg latency (s)")
        plt.title("Resolution Effect on Latency")
        plt.legend()
        pdf.savefig(); plt.close()

    print(f"PDF report written to {OUTPUT_PDF}")

if __name__ == "__main__":
    run_grid()
    make_pdf()