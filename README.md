# MobileSparrow

Lightweight human pose & object detection toolkit (MoveNet + SSDLite) for mobile/edge and server, built with PyTorch.

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fire717/Fire/blob/main/LICENSE)

Originally a MoveNet (17 keypoints) re-implementation, now extended to **MobileNet/ShuffleNet + FPN-Lite + SSDLite** for multi-object detection — all under a unified config/loader pipeline.

---

## 0) Dependencies (example)

You can use `pip install -r requirements.txt` to install necessary packages for this project.

```bash
pip install -r requirements.txt
````

---

## 1) COCO2017 Layout

Download COCO2017 (or use the helper script):

```bash
./scripts/get_coco2017.sh
```

When the zip packages are ready, it will unfold into the following structure:

```
data/
└── coco2017
    ├── annotations
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   ├── person_keypoints_train2017.json
    │   └── person_keypoints_val2017.json
    ├── train2017
    └── val2017
```

---

## 2) Data Preparation (lean scripts)

### 2.1 MoveNet (single-person square crops)

Uses official train/val only; filters instances with too few visible keypoints; crops a square that covers `bbox ∪ visible_keypoints` (with padding if needed).

```bash
python scripts/make_coco2017_for_movenet.py \
  --root ./data/coco2017 \
  --out-dir ./data/coco2017_movenet_sp \
  --splits train,val \
  --min-visible-kpts 8 \
  --expand-ratio 1.4 \
  --jpeg-quality 95
```

Output (plug-and-play with the loader):

```
data/coco2017_movenet_sp/
├── images/{train2017,val2017}/*.jpg
└── annotations/{person_keypoints_train2017.json, person_keypoints_val2017.json}
```

### 2.2 SSDLite (object detection)

Keeps official train/val; optional **class subset** filtering; skips crowd & tiny boxes; supports copy/symlink/none.

```bash
# keep five classes: person, car, bicycle, dog, chair
python scripts/make_coco2017_for_ssdlite.py \
  --root ./data/coco2017 \
  --out-dir ./data/coco2017_det_5cls \
  --class-names person,car,bicycle,dog,chair \
  --skip-crowd \
  --min-box-area 16
```

> You may also keep all classes at data stage and restrict classes **at train time** via `task_params.class_filter` (e.g., `[1]` for person-only).


---

**Notes**

* Older configs that used `num_classes: 17` for “number of joints” are supported by code-side bridging, but new configs should set `task_params.num_joints`.
* The loader now **expects `train2017` + `val2017` only** (no `test2017`). Wrong roots will error out early.

---

## 4) Train / Eval / Predict / Export

The `sparrow_cli.py` script provides a **unified CLI** for model training, evaluation, and export, all driven by **YAML configs**.
It supports **resume-from-checkpoint**, **safe instantiation via aliases or file paths**, and **export to ONNX/TorchScript**.

```bash
# Train
python sparrow_cli.py train -c configs/ssdlite.yaml

# Eval
python sparrow_cli.py eval -c configs/ssdlite.yaml

# Export
python sparrow_cli.py export -c configs/ssdlite.yaml
```

**YAML Example**

```yaml
seed: 42
deterministic: false

model:
  class: ssdlite
  args:
    num_classes: 81

trainer:
  class: dets_trainer
  args:
    epochs: 300
    save_dir: outputs/ssdlite_coco

data:
  train:
    builder: coco_dets_dataloader
    args:
      dataset_root: /path/to/coco
      img_size: 320
      batch_size: 64
  val:
    builder: coco_dets_dataloader
    args:
      dataset_root: /path/to/coco
      img_size: 320
      batch_size: 64
```


---

## 5) Layout (high level)

```
.
├── LICENSE
├── README.md
├── configs
│   ├── movenet_config.json
│   └── ssdlite_config.json
├── core
│   ├── datasets
│   │   ├── coco_dets.py
│   │   ├── coco_kpts.py
│   │   ├── coco_loader.py
│   │   ├── common.py
│   │   └── simple_loader.py
│   ├── loss
│   │   ├── movenet_loss.py
│   │   └── ssdlite_loss.py
│   ├── models
│   │   ├── backbones
│   │   │   ├── mobilenet_v2.py
│   │   │   └── shufflenet_v2.py
│   │   ├── heads
│   │   │   ├── movenet_head.py
│   │   │   └── ssd_head.py
│   │   ├── necks
│   │   │   ├── fpn_lite_dets.py
│   │   │   └── fpn_lite_kpts.py
│   │   ├── onnx
│   │   │   ├── dummy_movenet.py
│   │   │   └── dummy_ssdlite.py
│   │   ├── movenet.py
│   │   └── ssdlite.py
│   ├── task
│   │   ├── task_dets.py
│   │   └── task_kpts.py
│   └── utils
│       ├── logger.py
│       └── paths.py
├── movenet_cli.py
├── requirements.txt
├── scripts
│   ├── common.py
│   ├── get_coco2017.sh
│   ├── make_coco2017_for_movenet.py
│   └── make_coco2017_for_ssdlite.py
└── ssdlite_cli.py
```

## License

MIT (see the badge link above)