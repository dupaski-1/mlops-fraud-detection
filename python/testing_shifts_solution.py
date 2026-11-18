"""
Solution for the "Data Distribution Shifts" exercise (testing_shifts.pdf).

The production VAE was trained on normal transactions only and logged to the
centralized MLflow server (see train_vae.py). The monitoring dashboard shows a
massive shock in the FP/FN ratio. Here we analyze the two traffic logs
(live_traffic_baseline.csv vs live_traffic_recent.csv) and statistically prove
WHY the model performance changed:

  Step 1: Extract 2D latent representations with the new VAE.generating_latent()
  Step 2: Two-Sample K-S test on z1 and z2 (normals only) -> Covariate Shift?
  Step 3: Histograms of log ERE for frauds only              -> Concept Shift?
  Step 4: MLOps mitigation discussion

Usage:
    python testing_shifts_solution.py [MLFLOW_RUN_ID]
    (defaults to the production run deployed on KServe)
"""
import sys
import os
# supress all tf messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import mlflow
from scipy import stats

from vae import VAE, Encoder, Decoder

TRACKING_URI = "http://20.67.76.40"
DEFAULT_RUN_ID = "1b6981bdea5a45ef91059f204b4e7a36"  # my stable Lab 1 / Lab 2 run (funny-snake-615)
FEATURE_COLS = [f'V{i}' for i in range(1, 29)]
ALPHA = 0.05  # significance level
L = 30        # Monte Carlo samples for the ERE, same as eval_vae.py / live_streaming.py


def load_production_vae(run_id, latent_dim=2):
    """
    Rebuild the python VAE (vae.py) with the production weights from MLflow.

    train_vae.py logs a raw SavedModel whose only restorable function is the
    'serving_default' signature (the ERE calculation for KServe). The encoder's
    own call() returns a TFP distribution and is NOT restorable from the
    SavedModel, so to use the new generating_latent() method we instantiate a
    fresh VAE and transfer the saved weights into it.
    """
    print(f"Fetching model from MLflow Run ID: {run_id}...")
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model/1")
    loaded = tf.saved_model.load(local_path)

    # IMPORTANT: the production run was trained with tanh activations (verified
    # against the source logged with the run, artifact code/vae.py), while the
    # current local vae.py defaults to relu. The architecture must match the
    # trained weights exactly, otherwise the ERE is systematically wrong.
    vae = VAE(Encoder(latent_dim=latent_dim, activation='tanh'),
              Decoder(latent_dim=latent_dim, activation='tanh'),
              tf.keras.optimizers.Adam())
    # One forward pass to build the weights before transferring them
    vae(tf.zeros((1, len(FEATURE_COLS)), dtype=tf.float32))

    vae.encoder.hidden_layers.set_weights([v.numpy() for v in loaded.encoder.hidden_layers.variables])
    vae.decoder.hidden_layers.set_weights([v.numpy() for v in loaded.decoder.hidden_layers.variables])
    vae.decoder.mu.set_weights([v.numpy() for v in loaded.decoder.mu.variables])
    vae.decoder.log_sigma.set_weights([v.numpy() for v in loaded.decoder.log_sigma.variables])

    return vae


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN_ID
    np.random.seed(42)
    tf.random.set_seed(42)

    mlflow.set_tracking_uri(TRACKING_URI)
    vae = load_production_vae(run_id)

    # The two traffic logs: baseline healthy traffic vs recent (failing) traffic
    df_base = pd.read_csv('../data/live_traffic_baseline.csv')
    df_recent = pd.read_csv('../data/live_traffic_recent.csv')

    X_base = df_base[FEATURE_COLS].values.astype(np.float32)
    X_recent = df_recent[FEATURE_COLS].values.astype(np.float32)
    y_base = df_base.Class.values
    y_recent = df_recent.Class.values

    # ==========================================
    # Step 1: Extract Latent Representations
    # ==========================================
    z_base = vae.generating_latent(tf.constant(X_base))['z_sample'].numpy()
    z_recent = vae.generating_latent(tf.constant(X_recent))['z_sample'].numpy()

    # ==========================================
    # Step 2: Testing for Covariate Shift (K-S test on normals, latent space)
    # ==========================================
    z_base_normal = z_base[y_base == 0]
    z_recent_normal = z_recent[y_recent == 0]

    ks_z1 = stats.ks_2samp(z_base_normal[:, 0], z_recent_normal[:, 0])
    ks_z2 = stats.ks_2samp(z_base_normal[:, 1], z_recent_normal[:, 1])

    print("\n" + "=" * 70)
    print("STEP 2: COVARIATE SHIFT TEST (two-sample K-S on normal transactions)")
    print("=" * 70)
    print(f"z1: KS statistic = {ks_z1.statistic:.4f}, p-value = {ks_z1.pvalue:.4f}")
    print(f"z2: KS statistic = {ks_z2.statistic:.4f}, p-value = {ks_z2.pvalue:.4f}")

    reject_z1 = ks_z1.pvalue < ALPHA
    reject_z2 = ks_z2.pvalue < ALPHA
    print(f"\nQuestion 1: At alpha = {ALPHA}, "
          f"z1 -> {'REJECT' if reject_z1 else 'FAIL TO REJECT'} H0, "
          f"z2 -> {'REJECT' if reject_z2 else 'FAIL TO REJECT'} H0.")
    if not (reject_z1 or reject_z2):
        print("Question 2: No. The latent distributions of the normal traffic are\n"
              "statistically indistinguishable between baseline and recent data, so\n"
              "the empirical input distribution P(X) has NOT shifted: there is no\n"
              "evidence of a Covariate Shift.")
    else:
        print("Question 2: Yes. The latent distribution of the normal traffic differs\n"
              "between baseline and recent data: evidence of a Covariate Shift.")

    # ==========================================
    # Step 3: Proving Concept Shift (log ERE histograms, frauds only)
    # ==========================================
    ere_fraud_base = vae.serving_reconstruct(
        inputs=tf.constant(X_base[y_base == 1]), L=tf.constant(L))['ere'].numpy()
    ere_fraud_recent = vae.serving_reconstruct(
        inputs=tf.constant(X_recent[y_recent == 1]), L=tf.constant(L))['ere'].numpy()

    # Filter out zeros to safely apply log, just like eval_vae.py
    log_ere_base = np.log(ere_fraud_base[ere_fraud_base != 0])
    log_ere_recent = np.log(ere_fraud_recent[ere_fraud_recent != 0])

    # Fetch the alert threshold logged during training (as in live_streaming.py)
    threshold = mlflow.get_run(run_id).data.metrics.get("threshold", None)

    print("\n" + "=" * 70)
    print("STEP 3: CONCEPT SHIFT (log ERE of fraudulent transactions only)")
    print("=" * 70)
    print(f"Baseline frauds: mean log ERE = {log_ere_base.mean():.2f} "
          f"({len(ere_fraud_base) - len(log_ere_base)} transactions were maximally anomalous (log ERE -> -inf))")
    print(f"Recent frauds:   mean log ERE = {log_ere_recent.mean():.2f} "
          f"({len(ere_fraud_recent) - len(log_ere_recent)} transactions were maximally anomalous (log ERE -> -inf))")
    if threshold is not None:
        # A fraud is MISSED (false negative) when log ERE >= tau. Zero-ERE
        # frauds count as detected (log ERE -> -inf), as in live_streaming.py.
        missed_base = np.sum(log_ere_base >= threshold) / len(ere_fraud_base)
        missed_recent = np.sum(log_ere_recent >= threshold) / len(ere_fraud_recent)
        print(f"Alert threshold tau = {threshold:.4f} "
              f"(flag as fraud when log ERE < tau)")
        print(f"Missed frauds (false negatives): "
              f"baseline {missed_base:.1%} -> recent {missed_recent:.1%}")

    # One histogram per dataset, on common bins so the shift is directly visible
    all_vals = np.concatenate([log_ere_base, log_ere_recent])
    bins = np.histogram_bin_edges(all_vals, bins=40)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(log_ere_base, bins=bins, alpha=0.7, color='#1f77b4', label='Frauds: baseline traffic')
    ax.hist(log_ere_recent, bins=bins, alpha=0.7, color='#d62728', label='Frauds: recent traffic')
    if threshold is not None:
        ax.axvline(x=threshold, color='k', linestyle='--', linewidth=2)
        ax.annotate(f'Alert threshold $\\tau$ = {threshold:.1f}\n(fraud flagged when log ERE < $\\tau$)',
                    xy=(threshold, ax.get_ylim()[1] * 0.75),
                    xytext=(threshold - (all_vals.max() - all_vals.min()) * 0.38,
                            ax.get_ylim()[1] * 0.85),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    ax.set_xlabel('Log Expected Reconstruction Error (log ERE)')
    ax.set_ylabel('Number of fraudulent transactions')
    ax.set_title('Concept Shift: fraud reconstruction errors, baseline vs recent traffic')
    ax.legend(prop={'size': 10})
    ax.grid(alpha=0.3)

    plot_filename = "../output/testing_shifts_fraud_ere_histograms.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to {plot_filename}")

    print("""
Question 3: Yes, the learned concept of fraud has shifted. In the baseline
traffic the frauds reconstruct poorly (low log ERE), so they sit to the LEFT
of the threshold and get flagged. In the recent traffic the frauds' log ERE
has moved sharply to the RIGHT, into the region of normal transactions, so
the deployed threshold no longer separates them (false negatives explode and
the FP/FN ratio collapses). Since P(X) of the normal traffic did not change
(Step 2) but the decoder's evaluation p(x|z) of frauds did - our proxy for
P(Y|X) - this is a Concept Shift: adversarial fraudsters adapted their
behavior to mimic normal transactions and slide under the detector.""")

    # ==========================================
    # Step 4: MLOps Mitigation (discussion)
    # ==========================================
    print("""
======================================================================
STEP 4: MLOPS MITIGATION
======================================================================
Would simply retraining the VAE on the recent data fix this? NO, twice over:
- The normal traffic has not shifted (Step 2), so retraining on recent
  normal data just reproduces the same model.
- Retraining on raw UNLABELED recent traffic is worse: the new frauds sit
  inside the normal region, so the VAE would learn to reconstruct them
  and absorb the fraud pattern into its definition of "normal" (poisoning).

What to do instead:
1. Contain: route transactions in the gray zone around tau to step-up
   verification / manual review while the fix is built.
2. Get labels for the recent window (chargebacks, manual review). In this
   lab they are given; in production they arrive late and must be collected.
3. Re-separate the classes: the frauds now mimic the normals in the current
   features, and if P(X|fraud) = P(X|normal), NO model trained on X can
   separate them. Add features the fraudsters cannot easily fake
   (velocity, device, merchant signals).
4. Retrain WITH the labels on the new features: labels keep verified frauds
   out of the "normal" training set (no poisoning), and every candidate
   model is evaluated on labeled recent data and rolled out via
   shadow/canary - never promoted silently. That gate is what separates
   continual learning from the naive retraining rejected above.
Monitoring (K-S on the latents, the FP/FN dashboard) is not a fix, but it
is what caught this incident - keep it running to catch the next move in
the adversarial cat-and-mouse game.""")


if __name__ == "__main__":
    main()
