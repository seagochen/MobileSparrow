# MobileSparrow

Lightweight human pose & object detection toolkit (MoveNet + SSDLite) for mobile/edge and server, built with PyTorch.


[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fire717/Fire/blob/main/LICENSE)

Originally a MoveNet (17 keypoints) re-implementation, now extended to **MobileNet/ShuffleNet + FPN-Lite + SSDLite** for multi-object detection — all under a unified config/loader pipeline.

---

## ✨ What’s New

* **Modular model layout**: `backbone (MobileNetV2 / ShuffleNetV2)` + `neck (FPN-Lite)` + `head (MoveNet / SSDLite)`
* **Unified config & loader**: one JSON toggles `"task": "kpts" | "det" | "cls"`; aligned data augmentations across tasks
* **Lean data prep scripts**: keep **official COCO train/val splits** only (no random re-split); detection script supports class subset filtering
* **Two CLIs**: `movenet_cli.py` (pose) and `ssdlite_cli.py` (detection) with `train / eval / predict / export-onnx`
* **ONNX export (detection)**: export `cls_logits / bbox_regs / anchors` for portable post-processing (decode/NMS outside)
* **Stability fixes**: lazy-built detection heads now auto-move to the right device; loss logging avoids `requires_grad` warnings

---

## 0) Dependencies (example)

You can use `pip install -r requirements.txt` to install necessary packages for this project.

```bash
pip install -r requirements.txt
```

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
  --expand-ratio 1.0 \
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

> You may also keep all classes at data stage and restrict classes **at train time** via `task_params.class_filter` (e.g. `[1]` for person-only).

Output:

```
data/coco2017_det_5cls/
├── images/{train2017,val2017}/*.jpg
└── annotations/{instances_train2017.json, instances_val2017.json}
```

---

## 3) Unified Config (one JSON for different tasks)

```json
{
  "GPU_ID": "0",
  "random_seed": 42,
  "cfg_verbose": true,

  "task": "kpts",                         // "kpts" | "det" | "cls"
  "task_params": {
    "num_joints": 17,                     // kpts only
    "export_keypoints": true,             // kpts only
    "class_agnostic_nms": false,          // det only (inference)
    "class_filter": [],                   // det only (e.g., [1] => person-only; omit/[] => all)
    "cls_mode": "single_label"            // cls (future use)
  },

  "save_dir": "output/",
  "dataset_root_path": "./data/coco2017_movenet_sp",   // switch per task

  "backbone": "mobilenet_v2",
  "width_mult": 1.0,
  "img_size": 256,
  "target_stride": 4,                     // kpts only

  "use_color_aug": true,
  "use_flip": true,
  "use_rotate": true,
  "rotate_deg": 30.0,
  "use_scale": true,
  "scale_range": [0.75, 1.25],
  "gaussian_radius": 2,                   // kpts only
  "sigma_scale": 1.0,                     // kpts only
  "select_person": "largest",             // kpts only

  "pin_memory": true,
  "num_workers": 8,
  "batch_size": 64,
  "epochs": 100,
  "learning_rate": 0.00035,
  "optimizer": "Adam",
  "scheduler": "MultiStepLR-90,130-0.2",
  "weight_decay": 0.0001,
  "clip_gradient": 1.0,
  "log_interval": 10
}
```

**Notes**

* Older configs that used `num_classes: 17` for “number of joints” are supported by code-side bridging, but new configs should set `task_params.num_joints`.
* The loader now **expects `train2017` + `val2017` only** (no `test2017`). Wrong roots will error out early.

---

## 4) Train / Eval / Predict / Export

### 4.1 MoveNet CLI

```bash
# train
python movenet_cli.py --config configs/movenet_config.json train

# eval (simple proxy metrics on val)
python movenet_cli.py --config configs/movenet_config.json eval

# predict a folder and visualize
python movenet_cli.py --config configs/movenet_config.json predict --images ./demo --out ./vis_kpts

# export ONNX (with keypoint outputs; dummy wrapper supported)
python movenet_cli.py --config configs/movenet_config.json export-onnx --out output/movenet.onnx
```

### 4.2 SSDLite CLI

```bash
# train (set "task": "det" in config; point dataset_root_path to your detection subset)
python ssdlite_cli.py --config configs/ssdlite_config.json train

# eval (simple proxy metrics on val)
python ssdlite_cli.py --config configs/ssdlite_config.json eval --weights output/best.pt

# predict a folder and visualize
python ssdlite_cli.py --config configs/ssdlite_config.json predict --images ./demo --out ./vis_det

# export ONNX (3 outputs)
python ssdlite_cli.py --config configs/ssdlite_config.json export-onnx \
  --out output/ssdlite.onnx --dynamic --verify
```

**ONNX (detection - Planning)**

* Outputs:

  * `cls_logits`: `[B, N, C]` (includes background)
  * `bbox_regs` : `[B, N, 4]` (anchor deltas)
  * `anchors`   : `[N, 4]` (cx, cy, w, h in \[0, 1])
* Do decode & NMS on the deployment side for better portability.

---

## 5) Layout (high level)

```
.
├── LICENSE
├── README.md
├── configs
│   ├── classification_config.json
│   ├── movenet_config.json
│   └── ssdlite_config.json
├── core
│   ├── datasets
│   │   ├── coco_cls.py
│   │   ├── coco_det.py
│   │   ├── coco_kpts.py
│   │   ├── coco_loader.py
│   │   ├── common.py
│   │   └── simple_loader.py
│   ├── loss
│   │   ├── movenet_loss.py
│   │   └── ssd_loss.py
│   ├── models
│   │   ├── backbones
│   │   │   ├── mobilenet_v2.py
│   │   │   └── shufflenet_v2.py
│   │   ├── heads
│   │   │   ├── movenet_head.py
│   │   │   └── ssd_head.py
│   │   ├── movenet.py
│   │   ├── necks
│   │   │   └── fpn_lite.py
│   │   ├── onnx
│   │   │   ├── dummy_movenet.py
│   │   │   └── dummy_ssdlite.py
│   │   └── ssdlite.py
│   ├── task
│   │   ├── task_det.py
│   │   └── task_kpts.py
│   └── utils
│       └── paths.py
├── movenet_cli.py
├── output
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
