import requests
import numpy as np
import mlflow

# 0. Setup MLflow to fetch the dynamic threshold logged during training
run_id = "1b6981bdea5a45ef91059f204b4e7a36"
mlflow.set_tracking_uri("http://20.67.76.40")

print(f"Fetching threshold from MLflow Run ID: {run_id}...")
run = mlflow.get_run(run_id)
threshold = run.data.metrics.get("threshold", 0.0)
print(f"Loaded Threshold: {threshold:.4f}\n")

# 1. Dummy transaction: unusual values (0.65 everywhere) simulate a fraud
dummy_transaction = [0.65] * 28

# KServe expects the TensorFlow Serving JSON format ("inputs" format)
payload = {
    "inputs": {
        "inputs": [dummy_transaction],  # Needs batch dimension [1, 28]
        "L": 30                         # Scalar input (no batch dimension)
    }
}

# 2. Send the request to my KServe URL
url = "http://vae-fraud-detector.s2613280.108.141.139.122.nip.io/v1/models/vae-fraud-detector:predict"
print(f"Sending request to {url}...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    # 3. Parse the outputs (the model calculates the ERE internally)
    ere = response.json()["outputs"][0]

    # 4. Convert to log scale and check against the threshold
    ere_log = np.log(ere) if ere > 0 else -np.inf

    print("--- Results ---")
    print(f"Expected Reconstruction Error (ERE): {ere:.8f}")
    print(f"Log ERE: {ere_log:.4f}")
    print(f"Alert Threshold: {threshold:.4f}\n")

    if ere_log < threshold:
        print("ALERT: Fraudulent Transaction Detected!")
    else:
        print("Transaction Normal")
else:
    print(f"Failed with status code: {response.status_code}")
    print(response.text)
