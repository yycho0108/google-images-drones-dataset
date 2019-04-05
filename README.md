# Google Drones Dataset

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

## 5. Convert 

[1]: https://github.com/hardikvasa/google-images-download
[2]: https://github.com/chuanenlin/drone-net
[3]: https://github.com/pjreddie/darknet
[4]: archive/detector.c
