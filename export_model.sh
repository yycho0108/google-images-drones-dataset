#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=''
TF_MODELS_PATH="${HOME}/Repos/models/research/"
PYTHONPATH="$PYTHONPATH:$TF_MODELS_PATH"
PYTHONPATH="$PYTHONPATH:$TF_MODELS_PATH/slim/"

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH='./training_demo/pipeline.config'
#TRAINED_CKPT_PREFIX='./training_demo/training/model.ckpt-'
CKPT_LATEST="$(head ./training_demo/training/checkpoint -n 1 | awk '{print $2}')"
CKPT_LATEST="${CKPT_LATEST%\"}"
CKPT_LATEST="${CKPT_LATEST#\"}"
TRAINED_CKPT_PREFIX="./training_demo/training/${CKPT_LATEST}"
echo "USING CHECKPOINT : $TRAINED_CKPT_PREFIX"
EXPORT_DIR='/tmp/model'
PYFILE="${HOME}/Repos/models/research/object_detection/export_inference_graph.py"

CUDA_VISIBLE_DEVICES='' python ${PYFILE} \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
