import argparse
import contextlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from datetime import timedelta
import tensorflow as tf
import mlflow

from vae import VAE, Encoder, Decoder


def main():
    parser = argparse.ArgumentParser(description='Train VAE model')
    parser.add_argument('--disable-mlflow', action='store_true', help='Disable MLflow logging')
    parser.add_argument('--epochs', default=100, type=int, help='Epochs to train VAE')
    parser.add_argument('--latent_dim', default=2, type=int, help='Latent dimensionality')
    args = parser.parse_args()

    use_mlflow = not args.disable_mlflow

    # TODO: Read the credit card dataset from '../data/creditcard.csv' using pandas
    # data = ...
    data = pd.read_csv('../data/creditcard.csv')
    # TODO: Extract the 'Class' column values into a variable 'y'
    y = data['Class'].values

    # TODO: Extract the features into 'X' by dropping 'Class', 'Amount', and 'Time'
    X = data.drop(columns=['Class', 'Amount', 'Time']).values

    # TODO: Create X_anomalies (where y == 1) and X_regular (where y == 0)
    X_anomalies = X[y==1,:]
    X_regular = X[y==0,:]

    # We use these, in addition to all fraud cases, to test the model
    keep = 5000

    # TODO: Create a random sample of training indices from X_regular (size should be total regular minus keep)
    np.random.seed(42)
    idx_tr = np.random.choice(range(int(X_regular.shape[0])), X_regular.shape[0] - keep, replace=False)

    # TODO: Create the test indices (the remaining indices not in idx_tr)
    idx_te = list(set(list(range(int(X_regular.shape[0])))) - set(idx_tr))

    # TODO: Create X_tr (only regular transactions using idx_tr)
    X_tr = X_regular[idx_tr]

    # TODO: Create X_te (concatenate regular test transactions and ALL anomalies). Convert to float32.
    X_te = np.concatenate((X_regular[idx_te], X_anomalies), axis=0).astype(np.float32)

    # TODO: Create y_te (concatenate zeros for regular test cases and ones for anomalies)
    y_te = np.concatenate((np.zeros(len(idx_te)), np.ones(X_anomalies.shape[0])))

    latent_dim = args.latent_dim
    epochs = args.epochs
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    vae = VAE(Encoder(latent_dim=latent_dim),
              Decoder(latent_dim=latent_dim),
              optimizer
              )

    train_dataset = tf.data.Dataset.from_tensor_slices(X_tr.astype(np.float32)).shuffle(X_tr.shape[0]).batch(128)
    loss_metric = tf.keras.metrics.Mean()
    elbo_all = []

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=vae)
    manager = tf.train.CheckpointManager(checkpoint, "../output/ckpts", max_to_keep=1)

    if use_mlflow:
        mlflow.set_tracking_uri("http://20.67.76.40")

        # Set experiment name to the student's name so runs are organized in the MLflow UI
        # REPLACE 'student_name_here' WITH YOUR ACTUAL NAME OR ID
        experiment_name = "s2613280"
        mlflow.set_experiment(experiment_name)

    # We wrap the training process in MLflow to log metrics and images per epoch
    # If MLflow is disabled, we use a nullcontext so the 'with' block still works
    mlflow_context = mlflow.start_run() if use_mlflow else contextlib.nullcontext()

    with mlflow_context as run:
        if use_mlflow:
            print(f"--- Started MLflow Run ID: {run.info.run_id} ---")
            mlflow.log_param("latent_dim", latent_dim)
            mlflow.log_param("epochs", epochs)
        else:
            print("--- Running with MLflow DISABLED ---")

        start = time.time()
        while int(checkpoint.step) < epochs:
            for i, x_batch in enumerate(train_dataset):
                elbo = vae.train(x_batch)
                loss_metric(elbo)

            current_epoch = int(checkpoint.step) + 1
            current_elbo = loss_metric.result()
            elbo_all.append(current_elbo)

            # Log the ELBO metric to MLflow
            if use_mlflow:
                mlflow.log_metric("elbo", float(current_elbo), step=current_epoch)

            if current_epoch % 10 == 0:
                print('Epoch {}: ELBO {:.4f}'.format(current_epoch, current_elbo))
                qz_x = vae.encoder(X_te)
                z = qz_x.sample().numpy()
                plt.scatter(z[y_te == 0, 0], z[y_te == 0, 1], s=8, label='Regular', color='b')
                plt.scatter(z[y_te == 1, 0], z[y_te == 1, 1], s=8, label='Fraud', color='r')
                plt.grid()
                plt.legend()

                # Save the image instead of showing it, and log to MLflow
                plot_filename = f"../output/latent_space_epoch_{current_epoch}.png"
                plt.savefig(plot_filename)
                if use_mlflow:
                    mlflow.log_artifact(plot_filename)
                plt.close()

            checkpoint.step.assign_add(1)

        print('Total time: {} seconds'.format(timedelta(seconds=time.time() - start)))
        save_path = manager.save(checkpoint_number=int(checkpoint.step))
        print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))

        # --- EXPORT FOR KSERVE & LOG TO MLFLOW ---
        print("Exporting model to SavedModel (.pb) format for KServe...")

        # 1. Dynamically extract the exact input shape from the training data
        input_dim = X_tr.shape[1]

        # 2. Create the dynamic serving signature
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32)
        ])
        def serve_reconstruct(inputs, L):
            return vae.serving_reconstruct(inputs, L)

        # 3. Export to a local folder
        export_path = "../output/vae_saved_model"
        tf.saved_model.save(vae, export_path, signatures={"serving_default": serve_reconstruct})

        if use_mlflow:
            # 4. Log the entire folder containing the .pb file as an artifact
            mlflow.log_artifacts(export_path, artifact_path="model/1")

            # 5. Log all python code used to generate this run for full reproducibility
            mlflow.log_artifacts(".", artifact_path="code")

            print("Successfully logged run, .pb model artifact, and source code to centralized MLflow!")
            print(f"\n---> COPY THIS RUN ID FOR EVALUATION: {run.info.run_id} <---")
        else:
            print("Successfully saved .pb model artifact locally (MLflow logging disabled).")


if __name__ == "__main__":
    main()