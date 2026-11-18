"""
CANARY RELEASE - Watching the traffic split live
================================================

Same streaming logic as live_streaming.py, but now the endpoint is running
TWO models at once (see canaryTrafficPercent in your model-deploy.yaml):

    stable = the original VAE (2 hidden layers, 28 features V1..V28)
    canary = the new VAE      (1 hidden layer,  14 features V1..V14)

Knative decides, request by request, which model answers. The response does
not tell us which one it was, but the input shapes do:

    * we send V1..V28 to the endpoint
        - 200 OK          -> the STABLE model answered
        - 400 Bad Request -> the router picked the CANARY (it only accepts 14 features)
    * on a 400 we resend V1..V14 until the canary answers

Only the first attempt counts as the routing decision, so the traffic split
we measure is the real one. Each model is scored with its own MLflow threshold.

A live figure shows:
    top    : share of traffic served by the canary (should settle near the
             canaryTrafficPercent in your YAML)
    below  : recall, precision and FP/FN ratio of the endpoint as a whole,
             of the stable model alone and of the canary alone
             -> promote or roll back?

Usage (run from fraud_project/python):
    python canary_streaming.py                       # 500 random baseline rows
    python canary_streaming.py --rows 2000           # more fraud cases, less noisy recall
    python canary_streaming.py --rows 0 --data recent  # all rows of the recent traffic
"""
import os
import time
import argparse
import requests
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# The stable model expects V1..V28, the canary only the first 14
STABLE_COLS = [f'V{i}' for i in range(1, 29)]
CANARY_COLS = [f'V{i}' for i in range(1, 15)]

DATASETS = {
    "baseline": "../data/live_traffic_baseline.csv",
    "recent": "../data/live_traffic_recent.csv",
}


def fetch_mlflow_threshold(run_id, tracking_uri="http://20.67.76.40"):
    """Fetches the dynamic threshold logged by eval_vae.py."""
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Fetching threshold from MLflow Run ID: {run_id}...")
    run = mlflow.get_run(run_id)
    threshold = run.data.metrics.get("threshold", 0.0)
    print(f"Loaded Threshold: {threshold:.4f}\n")
    return threshold


def _payload(features):
    """Format payload exactly as required by TF Serving / KServe."""
    return {
        "inputs": {
            "inputs": [features],  # Batch dimension [1, 28] or [1, 14]
            "L": 30                # Scalar input
        }
    }


# ==========================================
# 1. One transaction: who answered, and what did it say?
# ==========================================
def score_transaction(session, endpoint_url, row, max_retries=50):
    """
    Returns (model, ere) where model is "stable" or "canary".

    A 400 on the 28-feature payload means the router sent us to the canary.
    We then resend the 14-feature payload; if that lands on the stable model
    it is rejected too, so we simply try again. Every retry is routed
    independently, so with a 20% canary all retries miss it with probability
    0.8**max_retries (~1% for 20 retries, ~0.001% for 50).
    """
    # First attempt: this is the routing decision we record
    response = session.post(endpoint_url, json=_payload(row[STABLE_COLS].tolist()))
    if response.status_code == 200:
        return "stable", response.json()["outputs"][0]
    if response.status_code != 400:
        response.raise_for_status()

    # The canary was picked: fetch its answer with the 14 features it expects
    canary_payload = _payload(row[CANARY_COLS].tolist())
    for _ in range(max_retries):
        response = session.post(endpoint_url, json=canary_payload)
        if response.status_code == 200:
            return "canary", response.json()["outputs"][0]
        if response.status_code != 400:
            response.raise_for_status()

    raise RuntimeError(f"Router picked the canary, but all {max_retries} retries with 14 features "
                       f"landed on the stable model. If this happens on every row, check that "
                       f"the canary really is a 14-feature model.")


# ==========================================
# 2. Live figure
# ==========================================
# Metrics plotted below the traffic split: (key, y-axis label)
METRICS = [
    ("recall", "Cumulative recall"),
    ("precision", "Cumulative precision"),
    ("fp_fn", "Cumulative FP/FN ratio"),
]
# One line per metric for the endpoint as a whole and for each model alone
SERIES = ["endpoint", "stable", "canary"]


def _metric_or_nan(metric, y_true, y_pred):
    """
    Each metric is undefined (NaN, a gap in the plot) until there is something to divide by:
        recall    -> at least one fraud case seen
        precision -> at least one transaction flagged as fraud
        fp_fn     -> at least one missed fraud case (FN > 0), same rule as eval_vae.py
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if metric == "recall":
        if y_true.sum() == 0:
            return np.nan
        return recall_score(y_true, y_pred, zero_division=0)
    if metric == "precision":
        if y_pred.sum() == 0:
            return np.nan
        return precision_score(y_true, y_pred, zero_division=0)
    # FP/FN ratio: false alarms per missed fraud case
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return fp / fn if fn > 0 else np.nan


def _setup_figure(title):
    plt.ion()
    fig, axes = plt.subplots(1 + len(METRICS), 1, figsize=(9, 12), sharex=True)
    ax_split = axes[0]
    fig.suptitle(title)

    ax_split.set_ylabel("Share served by canary (%)")
    ax_split.set_ylim(0, 100)
    ax_split.grid(alpha=0.3)

    lines = {"split": ax_split.plot([], [], color="tab:orange", label="canary share")[0]}
    styles = {
        "endpoint": dict(color="black", lw=2, label="endpoint (what customers see)"),
        "stable": dict(color="tab:blue", label="stable model"),
        "canary": dict(color="tab:orange", label="canary model"),
    }
    for ax, (metric, ylabel) in zip(axes[1:], METRICS):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        for series in SERIES:
            lines[f"{metric}_{series}"] = ax.plot([], [], **styles[series])[0]
        # Recall and precision live in [0, 1]; the FP/FN ratio has no upper bound
        if metric != "fp_fn":
            ax.set_ylim(0, 1.05)
    axes[-1].set_xlabel("Transactions processed")

    ax_split.legend(loc="upper right")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return fig, axes, lines


def _update_figure(fig, axes, lines, history):
    for key in lines:
        lines[key].set_data(history["n"], history[key])
    for ax in axes:
        ax.set_xlim(0, max(history["n"][-1], 1))
    # Rescale the FP/FN axis to the largest ratio seen so far (it has no fixed upper bound)
    ratios = np.array([history[f"fp_fn_{series}"] for series in SERIES], dtype=float)
    ymax = np.nanmax(ratios) if np.isfinite(ratios).any() else 1.0
    axes[-1].set_ylim(0, max(1.05 * ymax, 1.0))
    fig.canvas.draw_idle()
    plt.pause(0.01)


# ==========================================
# 3. Streaming
# ==========================================
def stream_canary(csv_path, endpoint_url, thresholds, delay_seconds=0.001,
                  seed=42, max_rows=500, plot_every=50, png_path=None):
    """
    Streams data row-by-row to the KServe endpoint, records which model
    answered each transaction and plots the canary split, recall, precision
    and FP/FN ratio live.

    thresholds: {"stable": float, "canary": float} from MLflow.
    """
    print(f"--- Starting Canary Stream for {csv_path} ---")
    df = pd.read_csv(csv_path)
    # Shuffle and reset the index: the dashboard below relies on a 0..N-1 index
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if max_rows:
        df = df.head(max_rows)

    routes, y_true, y_pred = [], [], []
    history = {"n": [], "split": []}
    history.update({f"{metric}_{series}": [] for metric, _ in METRICS for series in SERIES})

    fig, axes, lines = _setup_figure(f"Canary release - {os.path.basename(csv_path)}")
    session = requests.Session()
    start_time = time.time()

    for index, row in df.iterrows():
        try:
            model, ere = score_transaction(session, endpoint_url, row)
        except Exception as e:
            print(f"Request failed at index {index}: {e}")
            continue

        # Convert to Log Scale (Handle 0 safely)
        ere_log = np.log(ere) if ere > 0 else -np.inf

        # Anomalies have a log density strictly LESS than the model's own threshold
        predicted_fraud = 1 if ere_log < thresholds[model] else 0

        routes.append(model)
        y_true.append(int(row['Class']))
        y_pred.append(predicted_fraud)

        # Optional: Sleep to simulate realistic live stream pacing
        time.sleep(delay_seconds)

        # --- Live Dashboard Updates ---
        if (index + 1) % plot_every == 0 or (index + 1) == len(df):
            r = np.array(routes)
            yt, yp = np.array(y_true), np.array(y_pred)
            history["n"].append(len(r))
            history["split"].append(100 * np.mean(r == "canary"))
            masks = {"endpoint": np.ones(len(r), dtype=bool), "stable": r == "stable", "canary": r == "canary"}
            for metric, _ in METRICS:
                for series, mask in masks.items():
                    history[f"{metric}_{series}"].append(_metric_or_nan(metric, yt[mask], yp[mask]))
            _update_figure(fig, axes, lines, history)

            print(f"Processed: {index + 1}/{len(df)} | "
                  f"Canary share: {history['split'][-1]:5.1f}% | "
                  f"Recall endpoint: {history['recall_endpoint'][-1]:.4f} | "
                  f"stable: {history['recall_stable'][-1]:.4f} | canary: {history['recall_canary'][-1]:.4f}")

    # --- Final Summary ---
    end_time = time.time()
    print("\n" + "="*40)
    print(f"STREAM COMPLETE ({end_time - start_time:.2f} seconds)")
    print("="*40)

    r = np.array(routes)
    yt, yp = np.array(y_true), np.array(y_pred)
    for model in ["stable", "canary"]:
        mask = r == model
        cm = confusion_matrix(yt[mask], yp[mask], labels=[0, 1])
        print(f"{model.upper()} model: {mask.sum()} transactions "
              f"({100 * mask.mean():.1f}% of traffic, {yt[mask].sum()} fraud cases)")
        print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
        print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
        print(f"Precision: {precision_score(yt[mask], yp[mask], zero_division=0):.4f} | "
              f"Recall: {recall_score(yt[mask], yp[mask], zero_division=0):.4f} | "
              f"FP/FN: {_metric_or_nan('fp_fn', yt[mask], yp[mask]):.4f}")
        print("-" * 40)
    print("="*40 + "\n")

    if png_path:
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        fig.savefig(png_path, dpi=150)
        print(f"Figure saved to {png_path}")

    # Keep the window open until the student closes it
    plt.ioff()
    plt.show()


# ==========================================
# Example Usage for the CANARY RELEASE exercise
# ==========================================
if __name__ == "__main__":
    # 0. Command line: how many random rows, and which traffic to stream
    parser = argparse.ArgumentParser(description="Canary release: stream traffic and watch the split live")
    parser.add_argument("--rows", type=int, default=500,
                        help="random rows to stream; 0 means all rows (default: 500)")
    parser.add_argument("--data", choices=DATASETS.keys(), default="baseline",
                        help="which live traffic to stream (default: baseline)")
    args = parser.parse_args()
    max_rows = args.rows if args.rows > 0 else None
    tag = str(args.rows) if max_rows else "all"

    # 1. Setup MLflow and Endpoint configuration
    STABLE_RUN_ID = "1b6981bdea5a45ef91059f204b4e7a36"               # the model you deployed in Lab 1
    CANARY_RUN_ID = open("../canary_release/CANARY_RUN_ID.txt").read().strip()  # written by canary_release/run_canary.sh
    # Replace with the actual URL from `kubectl get inferenceservice`
    ENDPOINT_URL = "http://vae-fraud-detector.s2613280.108.141.139.122.nip.io/v1/models/vae-fraud-detector:predict"

    # 2. Each model is judged with its own threshold
    thresholds = {
        "stable": fetch_mlflow_threshold(STABLE_RUN_ID),
        "canary": fetch_mlflow_threshold(CANARY_RUN_ID),
    }

    # 3. Stream the traffic and watch the canary
    stream_canary(DATASETS[args.data], ENDPOINT_URL, thresholds, max_rows=max_rows,
                  png_path=f"../output/canary_{args.data}_{tag}.png")
