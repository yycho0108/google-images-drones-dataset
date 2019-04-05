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

```bash
./darknet detector test cfg/drone.data cfg/yolo-drone.cfg weights/yolo-drone.weights "/media/ssd/datasets/drones/all/"
```

[1]: https://github.com/hardikvasa/google-images-download
[2]: https://github.com/chuanenlin/drone-net
