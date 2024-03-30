#!/bin/bash

python train_test_runner.py \
  --telegram \
  --datasets_folder datasets \
  --losses "[l2,huber,huber_scheduled_backwards]" \
  --huber_deltas "[0.001]" \
  --learning_rates "[7e-6]" \
  --max_training_steps 2001 \
  --overwrite_results
#  --exit_on_fail
