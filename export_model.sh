#!/usr/bin/env bash
# TODO : accept arguments
AUTOPATH=1
INPUT_TYPE=image_tensor
EXPORT_DIR='/tmp/model'
TF_MODELS_PATH="${HOME}/Repos/models/research/"

if [ "$AUTOPATH" -eq "1" ]; then
    # Auto Path
    TRAIN_PATH="$(find /tmp/train -maxdepth 1 -printf '%T+ %p\n' | sort -r | head -n 1| cut -d ' ' -f2)"
    echo "TRAIN_PATH : ${TRAIN_PATH}"
    CKPT_LATEST="$(head ${TRAIN_PATH}/checkpoint -n 1 | awk '{print $2}')"
    CKPT_LATEST="${CKPT_LATEST%\"}"
    CKPT_LATEST="${CKPT_LATEST#\"}"
    TRAINED_CKPT_PREFIX="${TRAIN_PATH}/${CKPT_LATEST}"
    PIPELINE_CONFIG_PATH="${TRAIN_PATH}/pipeline.config"
else
    # Manual Path
    PIPELINE_CONFIG_PATH='/tmp/train/7/pipeline.config'
    TRAINED_CKPT_PREFIX="/tmp/train/7/model.ckpt-200000"
fi

echo "USING CHECKPOINT : $TRAINED_CKPT_PREFIX"

# Ensure directory exists
if [ -d "$EXPORT_DIR" ]; then
  # control will enter here if $EXPORT_DIR exists.
  echo "Export Directory: $EXPORT_DIR already exists!"
  rm -rfI $EXPORT_DIR
fi

# Finally, export
PYFILE="${TF_MODELS_PATH}/object_detection/export_inference_graph.py"
PYTHONPATH="$PYTHONPATH:$TF_MODELS_PATH:$TF_MODELS_PATH/slim/" python ${PYFILE} \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
