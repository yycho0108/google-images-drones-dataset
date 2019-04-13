#!/usr/bin/env bash

DATA_ROOT="/media/ssd/datasets/coco/"
IMG_ROOT="${DATA_ROOT}/raw-data/"
ANN_ROOT="${DATA_ROOT}/raw-data/annotations"
REC_ROOT="${DATA_ROOT}/record/"

TF_MODELS_PATH="${HOME}/Repos/models/research/"
PYFILE="${TF_MODELS_PATH}/object_detection/dataset_tools/create_coco_tf_record.py"

#bash ${TF_MODELS_PATH}/object_detection/dataset_tools/download_and_preprocess_mscoco.sh ${DATA_ROOT}

PYTHONPATH="$PYTHONPATH:$TF_MODELS_PATH" python ${PYFILE} --logtostderr \
  --train_image_dir="${IMG_ROOT}/train2017" \
  --val_image_dir="${IMG_ROOT}/val2017" \
  --test_image_dir="${IMG_ROOT}/test2017" \
  --train_annotations_file="${ANN_ROOT}/instances_train2017.json" \
  --val_annotations_file="${ANN_ROOT}/instances_val2017.json" \
  --testdev_annotations_file="${ANN_ROOT}/image_info_test-dev2017.json" \
  --output_dir="${REC_ROOT}"
