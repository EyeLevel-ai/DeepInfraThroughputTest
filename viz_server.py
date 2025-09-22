from flask import Flask, request, render_template_string
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

CSV_FILE = "grid_results.csv"

# Simple HTML template with filters
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Cost & Latency Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2rem; }
        form { margin-bottom: 2rem; }
        img { border: 1px solid #ccc; margin-top: 1rem; }
        label { margin-right: 0.5rem; }
        select { margin-right: 1rem; }
    </style>
</head>
<body>
    <h1>Cost & Latency Dashboard</h1>
    <form method="get">
        <label>X-axis:</label>
        <select name="xparam">
            {% for p in params %}
            <option value="{{p}}" {% if p==xparam %}selected{% endif %}>{{p}}</option>
            {% endfor %}
        </select>

        <label>Group by:</label>
        <select name="hueparam">
            <option value="">(none)</option>
            {% for p in params %}
            <option value="{{p}}" {% if p==hueparam %}selected{% endif %}>{{p}}</option>
            {% endfor %}
        </select>
        <br><br>

        {% for p in params %}
        <label>{{p}}:</label>
        <select name="filter_{{p}}">
            <option value="">(all)</option>
            {% for v in unique_values[p] %}
            <option value="{{v}}" {% if filters[p]==v|string %}selected{% endif %}>{{v}}</option>
            {% endfor %}
        </select>
        {% endfor %}

        <br><br>
        <button type="submit">Update</button>
    </form>

    {% if plot_cost %}
    <h2>Average Cost vs {{xparam}}</h2>
    <img src="data:image/png;base64,{{plot_cost}}" />
    <h2>Average Latency vs {{xparam}}</h2>
    <img src="data:image/png;base64,{{plot_latency}}" />
    {% endif %}
</body>
</html>
"""

def plot_metric(df, xparam, hueparam, metric, ylabel):
    fig, ax = plt.subplots(figsize=(8,6))

    if hueparam:
        for val, subset in df.groupby(hueparam):
            agg = subset.groupby(xparam)[metric].mean().reset_index()
            ax.plot(agg[xparam], agg[metric], marker="o", label=f"{hueparam}={val}")
        ax.legend()
    else:
        agg = df.groupby(xparam)[metric].mean().reset_index()
        ax.plot(agg[xparam], agg[metric], marker="o")

    ax.set_xlabel(xparam)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {xparam}")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/", methods=["GET"])
def index():
    df = pd.read_csv(CSV_FILE)
    params = ["input_tokens", "output_tokens", "resolution", "num_images"]

    # Build unique values for filters
    unique_values = {p: sorted(df[p].unique()) for p in params}

    # Current selections
    xparam = request.args.get("xparam", "input_tokens")
    hueparam = request.args.get("hueparam", "")

    filters = {}
    for p in params:
        filters[p] = request.args.get(f"filter_{p}", "")

    # Apply filters
    for p, val in filters.items():
        if val != "":
            # cast to correct dtype
            if df[p].dtype.kind in "if":  # numeric
                val = float(val)
                if val.is_integer():
                    val = int(val)
            df = df[df[p] == val]

    if df.empty:
        plot_cost = plot_latency = None
    else:
        plot_cost = plot_metric(df, xparam, hueparam, "cost", "Average Cost")
        plot_latency = plot_metric(df, xparam, hueparam, "latency", "Average Latency (s)")

    return render_template_string(
        TEMPLATE,
        params=params,
        xparam=xparam,
        hueparam=hueparam,
        plot_cost=plot_cost,
        plot_latency=plot_latency,
        unique_values=unique_values,
        filters=filters,
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
