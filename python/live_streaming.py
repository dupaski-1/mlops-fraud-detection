import requests
import numpy as np
import pandas as pd
import mlflow
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def fetch_mlflow_threshold(run_id, tracking_uri="http://20.67.76.40"):
    """Fetches the dynamic threshold logged during VAE training."""
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Fetching threshold from MLflow Run ID: {run_id}...")
    run = mlflow.get_run(run_id)
    threshold = run.data.metrics.get("threshold", 0.0)
    print(f"Loaded Threshold: {threshold:.4f}\n")
    return threshold

def stream_and_evaluate(csv_path, endpoint_url, threshold, delay_seconds=0.001, seed=42):
    """
    Streams data row-by-row to the KServe endpoint to simulate live traffic,
    records predictions, and prints a live performance dashboard.

    Rows are shuffled before streaming so fraud cases arrive at random points
    in the stream. Use a fixed seed for a reproducible order, or seed=None
    for a different order on every run.
    """
    print(f"--- Starting Live Stream for {csv_path} ---")
    df = pd.read_csv(csv_path)
    # Shuffle and reset the index: the dashboard below relies on a 0..N-1 index
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    y_true = []
    y_pred = []

    # We expect features V1 to V28 (28 PCA components)
    feature_cols = [f'V{i}' for i in range(1, 29)]

    start_time = time.time()

    for index, row in df.iterrows():
        # 1. Extract true label and features
        actual_fraud = int(row['Class'])
        features = row[feature_cols].tolist()

        # 2. Format payload exactly as required by TF Serving / KServe
        payload = {
            "inputs": {
                "inputs": [features],  # Batch dimension [1, 28]
                "L": 30                # Scalar input
            }
        }

        # 3. Send request
        try:
            # If a canary (14-feature model) is live on the same endpoint, the router sometimes
            # sends our 28-feature request to it and it answers 400. Resend until the stable
            # model answers, so this script always scores the stable model only.
            for _ in range(50):
                response = requests.post(endpoint_url, json=payload)
                if response.status_code != 400:
                    break
            response.raise_for_status() # Raise exception for bad status codes

            predictions = response.json()
            ere = predictions["outputs"][0]

            # 4. Convert to Log Scale (Handle 0 safely)
            if ere > 0:
                ere_log = np.log(ere)
            else:
                ere_log = -np.inf

            # 5. Predict Fraud based on threshold
            # Anomalies have a log density strictly LESS than the threshold
            predicted_fraud = 1 if ere_log < threshold else 0

            y_true.append(actual_fraud)
            y_pred.append(predicted_fraud)

        except Exception as e:
            print(f"Request failed at index {index}: {e}")
            continue

        # Optional: Sleep to simulate realistic live stream pacing
        time.sleep(delay_seconds)

        # --- Live Dashboard Updates ---
        # Print a rolling update every 500 transactions
        if (index + 1) % 500 == 0 or (index + 1) == len(df):
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)

            # Calculate FP and FN to track the ratio
            cm_rolling = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fp_rolling = cm_rolling[0][1]
            fn_rolling = cm_rolling[1][0]
            fp_fn_ratio = (fp_rolling / fn_rolling) if fn_rolling > 0 else float('inf')

            print(f"Processed: {index + 1}/{len(df)} transactions | "
                  f"Live Recall: {recall:.4f} | Live Precision: {precision:.4f} | "
                  f"FP/FN Ratio: {fp_fn_ratio:.2f}")

    # --- Final Summary ---
    end_time = time.time()
    print("\n" + "="*40)
    print(f"STREAM COMPLETE ({end_time - start_time:.2f} seconds)")
    print("="*40)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fp_final = cm[0][1]
    fn_final = cm[1][0]
    final_fp_fn_ratio = (fp_final / fn_final) if fn_final > 0 else float('inf')

    print("Final Confusion Matrix:")
    print(f"TN: {cm[0][0]} | FP: {fp_final}")
    print(f"FN: {fn_final} | TP: {cm[1][1]}")
    print("-" * 40)
    print(f"Final Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Final Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Final FP/FN Ratio: {final_fp_fn_ratio:.2f}")
    print("="*40 + "\n")

# ==========================================
# Example Usage for LAB 4
# ==========================================
if __name__ == "__main__":
    # 1. Setup MLflow and Endpoint configuration
    RUN_ID = "1b6981bdea5a45ef91059f204b4e7a36"
    # Replace with the actual URL from `kubectl get inferenceservice`
    ENDPOINT_URL = "http://vae-fraud-detector.s2613280.108.141.139.122.nip.io/v1/models/vae-fraud-detector:predict"

    # 2. Get the dynamic threshold
    dynamic_threshold = fetch_mlflow_threshold(RUN_ID)

    # 3. Stream Baseline Traffic 
    # Expected: High Precision and Recall
    stream_and_evaluate('../data/live_traffic_baseline.csv', ENDPOINT_URL, dynamic_threshold)

    # 4. Stream Concept Shift Traffic (Week 3 - Adversarial Fraudsters)
    # Expected: Dashboard will show Recall plummeting to near zero. False Negatives will spike.
    stream_and_evaluate('../data/live_traffic_recent.csv', ENDPOINT_URL, dynamic_threshold)
