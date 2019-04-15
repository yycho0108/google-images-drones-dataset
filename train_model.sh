#!/usr/bin/env bash
#python train.py --logtostderr
#--train_dir=training_demo/training7/
#--pipeline_config_path=pipeline.config
mkdir -p "/tmp/train"
IDX="$(ls /tmp/train | wc -l)"
mkdir -p "/tmp/train/${IDX}"
cp pipeline.config "/tmp/train/${IDX}"
TF_MODELS_PATH="${HOME}/Repos/models/research/"
python $TF_MODELS_PATH/object_detection/model_main.py \
    --model_dir "/tmp/train/${IDX}" \
    --pipeline_config_path './pipeline.config' \
