#!/bin/bash
# Trains the canary (1 hidden layer, 14 features), saves its Run ID, then evaluates it
# so its own threshold is logged to MLflow. Run from fraud_project/canary_release.
set -e
python train_vae.py 2>&1 | tee train_log.txt
RUN_ID=$(grep "COPY THIS RUN ID" train_log.txt | sed -E 's/.*: ([0-9a-f]{32}).*/\1/')
echo "$RUN_ID" > CANARY_RUN_ID.txt
echo "Canary Run ID: $RUN_ID"
python eval_vae.py "$RUN_ID"
