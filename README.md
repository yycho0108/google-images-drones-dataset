# Google Drones Dataset

Drone Detection Dataset from Google Images

## 1. Download Raw Images

See [google-images-download][1] for details.

```bash
googleimagesdownload -cf cfg.json
```

## 2. Filter Duplicates by Perceptual Hash

```bash
python unique.py
```

## 3. Bootstrapping : Create Initial Annotations

See [drone-net][2] for obtaining initial weights and parts of the dataset.

(Note that [darknet][3] must be compiled with an alternate version of [detector.c][4] to produce annotations.)

```bash
mkdir -p /tmp/det
./darknet detector test cfg/drone.data cfg/yolo-drone.cfg weights/yolo-drone.weights "/media/ssd/datasets/drones/all/"
```

The resultant annotations will be stored in `/tmp/det` as `.txt` files.

The format of the output `.txt` file is as follows:

```bash
filename    # full image path
cx cy w h p # text header, field names
cx cy w h p # one row per box, in relative coordinates. p=confidence in [0-1] interval
```

## 4. Manually Inspect and correct the annotations

First, record the responses:

```bash
python fix_ann.py [pos:=true]
```

The resultant responses will be saved in four different files:

```bash
ann_neg_idx.npy
ann_neg_lbl.npy
ann_pos_idx.npy
ann_pos_lbl.npy
```

Then apply the responses:

```bash
python apply_responses.py
```

This should store the results in `dataset/`

## 5. Convert To TFRecords

```bash
python tfrec.py
```

## 6. Training

```bash
python train.py --logtostderr --train_dir=training3/ --pipeline_config_path=pipeline-fpn.config
```

## 7. Export model

```bash
bash export_model.sh
```

## 8. Testing

```bash
python object_detection_tf.py
```

[1]: https://github.com/hardikvasa/google-images-download
[2]: https://github.com/chuanenlin/drone-net
[3]: https://github.com/pjreddie/darknet
[4]: archive/detector.c
