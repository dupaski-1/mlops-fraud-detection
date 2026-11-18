"""
LAB 3 - Drift monitoring with Evidently AI
==========================================

Same streaming logic as live_streaming.py, with one difference: instead of
keeping only the labels, we keep EVERY row we send to the endpoint (features),
EVERY answer we get back (log reconstruction error) and the true label.

That gives us a table of "what the model saw and what it answered", which is
exactly what Evidently needs:

    reference  = the scored BASELINE traffic  (model behaves well)
    current    = the scored RECENT traffic    (fraudsters adapted)

Evidently then compares the two tables and writes an HTML report with
  * data drift      : did the input features V1..V28 change?
  * prediction drift: did the model's output (ere_log) change?
  * model quality   : precision / recall, using the true labels.

Usage (run from fraud_project/python):
    python drift_monitoring.py              # 500 random rows per dataset (~1 min)
    python drift_monitoring.py --rows 2000  # 2000 rows, dashboard line every 500
    python drift_monitoring.py --rows 0     # all 5492 rows per dataset (~12 min)

  1. Streams the baseline once -> saves ../output/reference_scored_<rows>.csv
  2. Streams the recent traffic -> compares it against the reference and
                                   writes ../output/drift_report_<rows>.html
"""
import os
import time
import argparse
import requests
import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from evidently import Report, Dataset, DataDefinition, BinaryClassification
from evidently.presets import DataDriftPreset, ClassificationPreset

# We expect features V1 to V28 (28 PCA components)
FEATURE_COLS = [f'V{i}' for i in range(1, 29)]


def fetch_mlflow_threshold(run_id, tracking_uri="http://20.67.76.40"):
    """Fetches the dynamic threshold logged during VAE training."""
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Fetching threshold from MLflow Run ID: {run_id}...")
    run = mlflow.get_run(run_id)
    threshold = run.data.metrics.get("threshold", 0.0)
    print(f"Loaded Threshold: {threshold:.4f}\n")
    return threshold


# ==========================================
# 1. Streaming: same loop as live_streaming.py, but we keep the rows
# ==========================================
def stream_and_collect(csv_path, endpoint_url, threshold, delay_seconds=0.001,
                       seed=42, reference_df=None, max_rows=500):
    """
    Streams data row-by-row to the KServe endpoint to simulate live traffic
    and returns a DataFrame with, for every transaction:
        V1..V28          the features we sent
        ere_log          log reconstruction error returned by the model
        Class            true label
        predicted_fraud  our decision (1 if ere_log < threshold)

    max_rows: stream only this many RANDOM rows of the file (default 500, one
    dashboard line). Use a multiple of 500 to get one line per 500 rows, or
    max_rows=None to stream the whole file.

    If reference_df is given, the rolling dashboard also prints how many
    feature columns Evidently flags as drifted so far.
    """
    print(f"--- Starting Live Stream for {csv_path} ---")
    df = pd.read_csv(csv_path)
    # Shuffle and reset the index: the dashboard below relies on a 0..N-1 index
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if max_rows is not None:
        df = df.head(max_rows)
    print(f"Streaming {len(df)} transactions ({int(df['Class'].sum())} frauds)")

    records = []
    start_time = time.time()

    for index, row in df.iterrows():
        # 1. Extract true label and features
        actual_fraud = int(row['Class'])
        features = row[FEATURE_COLS].tolist()

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
            response.raise_for_status()  # Raise exception for bad status codes

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

            # 6. NEW: keep the whole observation for Evidently
            record = dict(zip(FEATURE_COLS, features))
            record["ere_log"] = ere_log
            record["Class"] = actual_fraud
            record["predicted_fraud"] = predicted_fraud
            records.append(record)

        except Exception as e:
            print(f"Request failed at index {index}: {e}")
            continue

        # Optional: Sleep to simulate realistic live stream pacing
        time.sleep(delay_seconds)

        # --- Live Dashboard Updates ---
        # Print a rolling update every 500 transactions
        if (index + 1) % 500 == 0 or (index + 1) == len(df):
            scored = pd.DataFrame(records)
            precision = precision_score(scored["Class"], scored["predicted_fraud"], zero_division=0)
            recall = recall_score(scored["Class"], scored["predicted_fraud"], zero_division=0)

            cm_rolling = confusion_matrix(scored["Class"], scored["predicted_fraud"], labels=[0, 1])
            fp_rolling = cm_rolling[0][1]
            fn_rolling = cm_rolling[1][0]
            fp_fn_ratio = (fp_rolling / fn_rolling) if fn_rolling > 0 else float('inf')

            line = (f"Processed: {index + 1}/{len(df)} transactions | "
                    f"Live Recall: {recall:.4f} | Live Precision: {precision:.4f} | "
                    f"FP/FN Ratio: {fp_fn_ratio:.2f}")

            if reference_df is not None:
                share = drifted_feature_share(reference_df, scored)
                line += f" | Drifted features: {share:.0%}"
            print(line)

    end_time = time.time()
    print(f"\nSTREAM COMPLETE ({end_time - start_time:.2f} seconds)\n")

    scored = pd.DataFrame(records)
    # Evidently cannot work with -inf; keep it as a missing value in the table.
    scored["ere_log"] = scored["ere_log"].replace(-np.inf, np.nan)
    return scored


# ==========================================
# 2. Evidently: compare current traffic against the reference
# ==========================================
def _data_definition():
    """Tells Evidently which columns are features and which are the labels."""
    return DataDefinition(
        numerical_columns=FEATURE_COLS + ["ere_log"],
        classification=[BinaryClassification(target="Class",
                                             prediction_labels="predicted_fraud")],
    )


def drifted_feature_share(reference_df, current_df):
    """Quick number for the live dashboard: share of V1..V28 flagged as drifted."""
    defn = DataDefinition(numerical_columns=FEATURE_COLS)
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current_df[FEATURE_COLS], data_definition=defn),
        Dataset.from_pandas(reference_df[FEATURE_COLS], data_definition=defn),
    )
    for metric in snapshot.dict()["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            return metric["value"]["share"]
    return float("nan")


def generate_drift_report(reference_df, current_df, html_path, drift_share=0.3):
    """
    Builds the Evidently report (data drift + prediction drift + model quality),
    saves it as HTML and prints the headline numbers.

    drift_share: fraction of columns that must drift before Evidently declares
    dataset-level drift. The default (0.5) is too strict for our scenario,
    where only the fraud rows (about 9% of the traffic) change.
    """
    defn = _data_definition()
    snapshot = Report([DataDriftPreset(drift_share=drift_share),
                       ClassificationPreset()]).run(
        Dataset.from_pandas(current_df, data_definition=defn),
        Dataset.from_pandas(reference_df, data_definition=defn),
    )
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    snapshot.save_html(html_path)

    print("=" * 40)
    print("EVIDENTLY SUMMARY (current vs reference)")
    print("=" * 40)
    for metric in snapshot.dict()["metrics"]:
        name, value = metric["metric_name"], metric["value"]
        if name.startswith("DriftedColumnsCount"):
            print(f"Drifted columns      : {int(value['count'])} ({value['share']:.0%})")
        elif name.startswith("ValueDrift(column=ere_log"):
            print(f"Prediction drift     : {name.split('method=')[1].split(',')[0]} = {value:.4g}")
        elif name.startswith("Precision("):
            print(f"Precision            : {value:.4f}")
        elif name.startswith("Recall("):
            print(f"Recall               : {value:.4f}")
    print("-" * 40)
    print(f"Full report saved to: {html_path}")
    print("=" * 40 + "\n")
    return snapshot


# ==========================================
# Example Usage for LAB 3
# ==========================================
if __name__ == "__main__":
    # 0. Command line: how many random rows of each dataset to stream
    #    python drift_monitoring.py              -> 500 rows (quick test, 1 dashboard line)
    #    python drift_monitoring.py --rows 2000  -> 2000 rows (4 dashboard lines)
    #    python drift_monitoring.py --rows 0     -> the whole dataset (5492 rows)
    parser = argparse.ArgumentParser(description="Lab 3: stream traffic and build an Evidently drift report")
    parser.add_argument("--rows", type=int, default=500,
                        help="random rows to stream from each dataset; 0 means all rows (default: 500)")
    args = parser.parse_args()
    max_rows = args.rows if args.rows > 0 else None
    tag = str(args.rows) if max_rows else "all"

    # 1. Setup MLflow and Endpoint configuration
    RUN_ID = "1b6981bdea5a45ef91059f204b4e7a36"
    # Replace with the actual URL from `kubectl get inferenceservice`
    ENDPOINT_URL = "http://vae-fraud-detector.s2613280.108.141.139.122.nip.io/v1/models/vae-fraud-detector:predict"

    # One reference per sample size, so a 500-row test never mixes with a full run
    REFERENCE_CSV = f"../output/reference_scored_{tag}.csv"
    REPORT_HTML = f"../output/drift_report_{tag}.html"

    # 2. Get the dynamic threshold
    dynamic_threshold = fetch_mlflow_threshold(RUN_ID)

    # 3. Stream Baseline Traffic and save it as the REFERENCE
    # Expected: High Precision and Recall. Run this once.
    if not os.path.exists(REFERENCE_CSV):
        reference = stream_and_collect('../data/live_traffic_baseline.csv', ENDPOINT_URL, dynamic_threshold,
                                       max_rows=max_rows)
        os.makedirs(os.path.dirname(REFERENCE_CSV), exist_ok=True)
        reference.to_csv(REFERENCE_CSV, index=False)
        print(f"Reference saved to {REFERENCE_CSV}\n")
    else:
        reference = pd.read_csv(REFERENCE_CSV)
        print(f"Loaded existing reference from {REFERENCE_CSV} ({len(reference)} rows)\n")

    # 4. Stream Concept Shift Traffic (Week 3 - Adversarial Fraudsters)
    # Expected: Recall drops, the share of drifted features climbs, and the
    # report shows the model output (ere_log) moving towards "normal" values.
    current = stream_and_collect('../data/live_traffic_recent.csv', ENDPOINT_URL, dynamic_threshold,
                                 reference_df=reference, max_rows=max_rows)

    # 5. Generate the Evidently report
    generate_drift_report(reference, current, REPORT_HTML)
