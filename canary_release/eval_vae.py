import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tensorflow as tf
import mlflow

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_vae.py <MLFLOW_RUN_ID>")
        sys.exit(1)

    run_id = sys.argv[1]
    mlflow.set_tracking_uri("http://20.67.76.40")

    print(f"Fetching model from MLflow Run ID: {run_id}...")

    # 1. Download model from the Azure Artifact Store
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model/1")
    vae = tf.saved_model.load(local_path)
    serve_fn = vae.signatures['serving_default']

    # 2. Re-create the test dataset
    data = pd.read_csv('../data/creditcard.csv')
    y = data.Class.values
    X = data.drop(columns=['Class', 'Amount', 'Time']).values

    # canary: get only the first 14 features (V1..V14)
    X = X[:, :14]
    X_anomalies = X[y == 1, :]
    X_regular = X[y == 0, :]

    keep = 5000
    np.random.seed(42)
    idx_tr = np.random.choice(range(int(X_regular.shape[0])), X_regular.shape[0] - keep, replace=False)
    idx_te = list(set(list(range(X_regular.shape[0]))) - set(idx_tr))

    X_te = np.concatenate((X_regular[idx_te], X_anomalies), axis=0).astype(np.float32)
    y_te = np.concatenate((np.zeros(len(idx_te)), np.ones(X_anomalies.shape[0])))

    # 3. Use the KServe signature we exported
    ERE = serve_fn(inputs=tf.constant(X_te), L=tf.constant(30))['ere'].numpy()

    normal = ERE[y_te == 0]
    anomalies = ERE[y_te == 1]
    target = 0.70

    sample = np.log(anomalies[anomalies != 0])
    ecdf = sm.distributions.ECDF(sample)
    x_val = np.linspace(min(sample), max(sample), 500)
    y_ecdf = ecdf(x_val)

    threshold = x_val[np.count_nonzero(y_ecdf[y_ecdf <= target]) - 1]
    normal_log = np.log(normal[normal != 0])
    regular_flagged = np.sum(normal_log < threshold)

    fraud_not_flagged = np.sum(sample >= threshold)

    if fraud_not_flagged > 0:
        fp_fn_ratio = regular_flagged / fraud_not_flagged
    else:
        fp_fn_ratio = float('inf')

    # Save visualization
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(normal_log, bins=50, label='Regular', alpha=0.7)
    ax1.hist(sample, bins=30, color='red', label='Fraud', alpha=0.7)
    ax1.axvline(x=threshold, color='k', linestyle='--', linewidth=2)
    ax1.annotate('Threshold to detect {}% of fraud transactions\nat the cost of {} regular transactions (False Positives)'
                 .format(target * 100, regular_flagged),
                 xy=(threshold - 1, 150), xytext=(-10, 250),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    ax1.annotate('Flag as fraud attempt', xy=(0.28, 0.85), xytext=(0.28, 0.90), xycoords='axes fraction',
                 ha='center', va='bottom',
                 arrowprops=dict(arrowstyle='-[, widthB=9, lengthB=1.5', lw=2.0))
    ax1.legend()
    plot_filename = "../output/evaluation_histogram.png"
    plt.savefig(plot_filename)

    # 4. Resume the exact MLflow run and append results
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("eval_target", target)
        mlflow.log_metric("threshold", float(threshold))
        mlflow.log_metric("false_positives_count", float(regular_flagged))
        mlflow.log_metric("false_negatives_count", float(fraud_not_flagged))
        mlflow.log_metric("fp_fn_ratio", float(fp_fn_ratio))
        mlflow.log_artifact(plot_filename)

    print("Evaluation complete! Check the MLflow UI.")

if __name__ == "__main__":
    main()
