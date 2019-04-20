#!/usr/bin/env bash
AUTOPATH=0

TF_MODELS_PATH="${HOME}/Repos/models/research/"
PYTHONPATH="$PYTHONPATH:$TF_MODELS_PATH"
PYTHONPATH="$PYTHONPATH:$TF_MODELS_PATH/slim/"
#TRAIN_PATH='./training_demo/training6'
TRAIN_PATH='/tmp/train/5'

INPUT_TYPE=image_tensor
#PIPELINE_CONFIG_PATH="${TRAIN_PATH}/pipeline.config"
PIPELINE_CONFIG_PATH='/tmp/pipeline.config'

if [ "$AUTOPATH" -ne "0" ]; then
    CKPT_LATEST="$(head ${TRAIN_PATH}/checkpoint -n 1 | awk '{print $2}')"
    CKPT_LATEST="${CKPT_LATEST%\"}"
    CKPT_LATEST="${CKPT_LATEST#\"}"
    TRAINED_CKPT_PREFIX="${TRAIN_PATH}/${CKPT_LATEST}"
else
    TRAINED_CKPT_PREFIX="/tmp/model.ckpt-24519"
fi

echo "USING CHECKPOINT : $TRAINED_CKPT_PREFIX"
EXPORT_DIR='/tmp/model'

if [ -d "$EXPORT_DIR" ]; then
  # Control will enter here if $DIRECTORY exists.
  rm -rfI $EXPORT_DIR
fi

PYFILE="${TF_MODELS_PATH}/object_detection/export_inference_graph.py"

python ${PYFILE} \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
