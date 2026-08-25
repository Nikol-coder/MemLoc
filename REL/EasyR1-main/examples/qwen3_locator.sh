#!/bin/bash

set -x

export WANDB_MODE=offline

MODEL_PATH=Qwen3-8B  # replace it with your local file path


python3 -m verl.trainer.main \
    config=examples/config_locator_rl.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_locator_rl
