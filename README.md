# MLOps fraud detection: from VAE to canary release

Course project for MST 0051 Machine Learning Operations at BI Norwegian Business School (Autumn 2026).

A variational autoencoder (VAE) is trained on normal credit card transactions only. A transaction is flagged as fraud when its reconstruction error is unusually high. The model is then deployed, monitored and released in stages.

## What is in this repository

| Folder | Content |
|---|---|
| `python/` | Training, evaluation, deployment YAML generation, live streaming, drift monitoring, distribution shift test |
| `canary_release/` | A smaller challenger VAE (14 features, 1 hidden layer) and the script that trains it |
| `deploy/` | KServe InferenceService YAML, deployed with GitOps (ArgoCD) |
| `output/` | Plots, the Evidently drift report, saved model and a full write up of results (`async_results_and_answers.md`) |
| `data/` | Live traffic samples used to simulate production requests |

## The steps

1. Data audit: 3 kinds of leakage found and fixed, class imbalance handled.
2. Training: VAE with MLflow tracking, threshold chosen to catch 70% of the frauds.
3. Deployment: KServe on Kubernetes, with the YAML merged through a pull request.
4. Live streaming: baseline traffic against shifted traffic.
5. Distribution shifts: K-S test in the latent space, concept shift confirmed.
6. Monitoring: Evidently reports for feature, prediction and quality drift.
7. Canary release: 20% of the traffic to the challenger model, compared with the stable model.

## Main results

- On baseline traffic the stable model catches about 76% of the frauds.
- On shifted traffic both models fall to about 24%, because the frauds now look like normal transactions (concept shift).
- Decision: keep the stable model. The canary was not better on either file.

The full numbers and discussion are in `output/async_results_and_answers.md`.

## Data

The full dataset `creditcard.csv` is not included because it is larger than GitHub's 100 MB file limit. Download it from Kaggle (Credit Card Fraud Detection, ULB Machine Learning Group) and place it in `data/`.

## Note

The servers (MLflow tracking server and Kubernetes cluster) belonged to the course and were shut down after the course period, so the endpoint URLs and run IDs in the scripts no longer resolve.
