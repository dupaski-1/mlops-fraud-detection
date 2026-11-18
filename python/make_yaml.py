"""
Writes <fraud_project>/deploy/s2613280/model-deploy.yaml for the GitOps Pull Request.

    python make_yaml.py                   # Lab 2: stable model only
    python make_yaml.py --canary          # canary release: 20% to the run in ../canary_release/CANARY_RUN_ID.txt

It asks MLflow for the experiment ID and checks that the run really has the
model/1 artifact and a threshold metric, so the storageUri cannot be mistyped.
"""
import argparse
import os
import mlflow

STUDENT_ID = "s2613280"
STABLE_RUN_ID = "1b6981bdea5a45ef91059f204b4e7a36"
AZURE_PREFIX = "https://mlopsartifacts2026.blob.core.windows.net/mlflow-artifacts"

parser = argparse.ArgumentParser()
parser.add_argument("--canary", action="store_true")
args = parser.parse_args()

mlflow.set_tracking_uri("http://20.67.76.40")
client = mlflow.tracking.MlflowClient()

if args.canary:
    run_id = open("../canary_release/CANARY_RUN_ID.txt").read().strip()
else:
    run_id = STABLE_RUN_ID

run = client.get_run(run_id)
experiment_id = run.info.experiment_id
files = [f.path for f in client.list_artifacts(run_id, "model/1")]
print(f"Run {run_id} | experiment {experiment_id} | name {run.info.run_name}")
print(f"Artifacts in model/1: {files}")
print(f"Threshold metric: {run.data.metrics.get('threshold')}")
assert any(f.endswith("saved_model.pb") for f in files), "No saved_model.pb under model/1"
assert "threshold" in run.data.metrics, "Run eval_vae.py on this run first (no threshold metric)"

canary_line = "    canaryTrafficPercent: 20\n" if args.canary else ""
yaml = f'''apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "vae-fraud-detector"
  # MUST match your student ID
  namespace: "{STUDENT_ID}"
spec:
  predictor:
{canary_line}    serviceAccountName: kserve-sa
    minReplicas: 1
    maxReplicas: 1
    tolerations:
    - key: "reservedFor"
      operator: "Equal"
      value: "lab2pool"
      effect: "NoSchedule"
    nodeSelector:
      agentpool: "lab2pool"
    model:
      modelFormat:
        name: tensorflow
      # You MUST specify resource limits, otherwise your deployment will be rejected by the namespace quota
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
        limits:
          cpu: "1"
          memory: "2Gi"
      storageUri: "{AZURE_PREFIX}/{experiment_id}/{run_id}/artifacts/model"
'''
out_dir = f"../deploy/{STUDENT_ID}"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/model-deploy.yaml", "w") as f:
    f.write(yaml)
print("\n" + yaml)
print(f"Written to {out_dir}/model-deploy.yaml")
