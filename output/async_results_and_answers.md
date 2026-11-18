# MST 0051 asynchronous work: results and answers (frozen 18 Nov 2026)

## 1. Where everything lives
- Code and outputs on the Mac: ~/Desktop/MST 0051/LABS/LAB2/fraud_project/ (folders python, canary_release, data, output, deploy)
- GitHub: pull request #30 (Lab 2, stable model) and pull request #31 (canary release) in BI-DS/MST0051, opened from the fork dupaski-1/MST0051. Both merged on 18 Nov 2026.
- MLflow (http://20.67.76.40), experiment s2613280, experiment id 31:
  - funny-snake-615, run 1b6981bdea5a45ef91059f204b4e7a36: stable model (2 hidden layers, 28 features), threshold -58.6408
  - nosy-ox-862, run 3c8be743969e419f824403882e78cd49: canary model (1 hidden layer, 14 features), threshold -38.4501
- Endpoint (server wiped after 19 Nov): http://vae-fraud-detector.s2613280.108.141.139.122.nip.io/v1/models/vae-fraud-detector:predict

## 2. The story in 7 steps
1. Data audit: a dirty pipeline looked perfect because of 3 leaks. Fixed pipeline gives F1 0.0000, the honest score.
2. Imbalance: only 0.17% of transactions are fraud, so accuracy is misleading. Recall, precision and PR-AUC are used instead.
3. Lab 1: a VAE trained on normal transactions only. A fraud reconstructs badly, so its log ERE (expected reconstruction error) falls below a threshold. The threshold is chosen to catch 70% of the frauds. Everything is logged to MLflow.
4. Lab 2: the model is deployed with a KServe InferenceService YAML. The YAML goes to GitHub as a pull request, and ArgoCD copies it to the Kubernetes cluster (GitOps).
5. Shifts homework: the recent traffic shows a concept shift, not a covariate shift.
6. Lab 3: Evidently compares scored baseline traffic (reference) against scored recent traffic (current).
7. Canary: a smaller model gets 20% of the traffic through canaryTrafficPercent: 20, and is compared with the stable model before a promote or roll back decision.

## 3. Data audit homework
- The 3 leaks: target leakage (Account_Blocked_Next_Day is only known after the event), temporal leakage (random split mixed past and future), transformation leakage (scaler fitted before the split).
- Fix: drop the leaking column, sort by Time and split without shuffling, fit the scaler on the training rows only.
- Result on creditcard_dirty.csv: F1 0.0000 (Amount and Time alone cannot separate fraud).
- Imbalance results (test set, fraud class):

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Baseline | 0.0000 | 0.0000 | 0.0000 | 0.0024 |
| Weighted | 0.0025 | 0.2435 | 0.0050 | 0.0024 |
| Undersampled | 0.0026 | 0.2522 | 0.0051 | 0.0024 |
| SMOTE | 0.0026 | 0.2522 | 0.0052 | 0.0024 |

## 4. Shifts homework (testing_shifts)
- Two-sample K-S test on the normal transactions in the latent space: z1 p-value 0.9992, z2 p-value 0.9945. At alpha 0.05 we fail to reject H0, so there is no covariate shift.
- Mean log ERE of the frauds: baseline -61.75, recent -45.25. The frauds moved towards the normal region.
- Missed frauds at the threshold: 23.4% on baseline, 75.8% on recent. This is a concept shift: P(X) of the normals is unchanged, but the frauds now look normal.
- Mitigation: retraining on unlabeled recent data would teach the VAE to treat the new frauds as normal. Instead get labels, add features the fraudsters cannot fake, retrain with labels, and release through shadow or canary.

## 5. Lab 2 and live streaming (live_streaming.py, stable model only)

| | TN | FP | FN | TP | Precision | Recall | FP/FN |
|---|---|---|---|---|---|---|---|
| Baseline (5492 rows) | 4968 | 32 | 116 | 376 | 0.9216 | 0.7642 | 0.28 |
| Recent (5492 rows) | 4968 | 32 | 371 | 121 | 0.7908 | 0.2459 | 0.09 |

Same 32 false alarms in both files. Only the frauds changed.

## 6. Lab 3 (Evidently, 2000 random rows per file)
- Streaming recall: 0.7457 on the reference, 0.2365 on the current data. Precision 0.9348 and 0.8889.
- Drifted columns: 15 (48%). The drift_share was lowered to 30%, so dataset drift is declared. With 2000 rows (more than 1000) Evidently uses the normed Wasserstein distance, not K-S.
- Prediction drift on ere_log: Wasserstein (normed) 0.05314, below the 0.1 limit, so the prediction distribution barely moved even though recall collapsed. Quality metrics (recall) were the clearest signal.
- Only about 9% of rows (the frauds) changed, which dilutes feature drift. Monitor all 3 layers: features, predictions, quality.
- Report: output/drift_report_2000.html

## 7. Canary release (canary_streaming.py, all 5492 rows per file)

Baseline traffic:

| Model | Share | Fraud cases | TN | FP | FN | TP | Precision | Recall | FP/FN |
|---|---|---|---|---|---|---|---|---|---|
| Stable | 79.9% | 398 | 3963 | 26 | 95 | 303 | 0.9210 | 0.7613 | 0.2737 |
| Canary | 20.1% | 94 | 1006 | 5 | 27 | 67 | 0.9306 | 0.7128 | 0.1852 |

Recent (shifted) traffic:

| Model | Share | Fraud cases | TN | FP | FN | TP | Precision | Recall | FP/FN |
|---|---|---|---|---|---|---|---|---|---|
| Stable | 80.4% | 401 | 3993 | 23 | 305 | 96 | 0.8067 | 0.2394 | 0.0754 |
| Canary | 19.6% | 91 | 979 | 5 | 69 | 22 | 0.8148 | 0.2418 | 0.0725 |

Figures: output/canary_baseline_all.png and output/canary_recent_all.png. The measured split settled near 20%, as set in the YAML.

## 8. Discussion answers

**1. On the baseline data, is the difference in recall larger than chance? How many fraud cases did the canary see?**
The canary saw 94 fraud cases. Its recall is 0.7128 against 0.7613 for the stable model, a gap of 0.049. With 94 cases the standard error of the canary recall is about 0.047, and the standard error of the gap is about 0.051. The gap is smaller than 1 standard error (z about 0.9, p about 0.34), so it is well within what chance alone produces. We cannot say the canary is worse.

**2. On the recent data, which model degrades more? Would you promote the canary?**
Both collapse. The stable recall falls from 0.761 to 0.239 (a drop of 0.52). The canary recall falls from 0.713 to 0.242 (a drop of 0.47), and on the recent data the 2 models are identical within noise (0.2394 against 0.2418). I would not promote the canary: it is not better than the stable model on either file, and it does nothing about the concept shift. The real fix is labels and new features, not a swap of models. The only argument for promoting is cost, because the canary is smaller (14 features, 1 hidden layer) and so cheaper to run.

**3. Compare precision and FP/FN. Could a model with lower recall but far fewer false alarms still be a good business case?**
On baseline traffic the canary has slightly higher precision (0.931 against 0.921) and a lower FP/FN ratio (0.185 against 0.274), at the price of lower recall (0.713 against 0.761). It raises fewer false alarms per missed fraud. That can be a good business case if a false alarm is expensive (blocked cards, angry customers, manual review) and a missed fraud is cheap. To decide we would need the average loss of 1 missed fraud (the fraudulent amount) against the cost of 1 false alarm (review time and customer churn). That is a business decision, not something a metric decides.

**4. What does the endpoint (black) line tell you about the cost of a 20% canary against a direct replacement?**
The endpoint is a traffic weighted mix: 0.8 x 0.761 + 0.2 x 0.713 is about 0.752 recall on baseline (measured: 370 of 492 frauds caught, 0.752). Customers lose about 0.01 of recall, while the canary alone is 0.049 lower. A weaker new model therefore costs only 20% of its weakness. Direct replacement would expose 100% of the traffic to the full 0.049 loss, with no chance to watch first or to roll back before the damage.

**5. How would a shadow deployment change this experiment?**
In a shadow deployment both models score every request, and only the stable model's answer reaches the customer. The 2 models would be compared on exactly the same rows instead of 2 different random subsets, so the comparison is less noisy (all 492 frauds for each model, not 94 for the canary). No customer is exposed to the new model. The price is double the compute, and no real traffic effects (such as customer reactions) can be observed.

**Decision:** keep the stable model (roll back the canary by setting canaryTrafficPercent: 0, or reverting the storageUri to the stable run ID). Removing canaryTrafficPercent alone would promote the canary to 100%.

## 9. Exam notes
- Canary release: reduces risk by routing a small share to a new model next to the old one. Knative keeps 2 revisions, KServe splits the traffic.
- Shadow deployment: new model scores everything, answers hidden. A/B test: random routing to compare hypotheses, canary is for risk, not for hypothesis testing.
- Covariate shift: P(X) changes. Concept shift: P(Y|X) changes. Label shift: P(Y) changes.
- Monitoring layers: features (no labels needed), predictions (no labels needed), quality metrics (labels needed, but most direct).
- Evidently picks K-S below 1000 reference rows and normed Wasserstein above 1000.
- GitOps: the YAML in GitHub is the source of truth, ArgoCD syncs it to the cluster.
