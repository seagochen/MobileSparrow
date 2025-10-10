# MobileSparrow

Lightweight multi-task computer vision toolkit for mobile/edge and server deployment, built with PyTorch.

[![license](https://img.shields.io/badge/license-GPL%20v3-blue.svg)](https://github.com/fire717/Fire/blob/main/LICENSE)

A unified PyTorch framework featuring **three core vision tasks** optimized for mobile and edge devices:

- **Object Detection** – SSDLite with MobileNet/ShuffleNet backbone + FPN-Lite for real-time multi-object detection
- **Body Pose Estimation**<sup>1</sup> – MoveNet (17 keypoints) for single-person pose tracking with sub-pixel accuracy
- **Head Rotation Estimation** – 6D rotation representation (SixDRepNet) for precise head pose estimation

All models share a common architecture pattern: **lightweight backbones + FPN + task-specific heads**, unified under a single config/loader/trainer pipeline.

---



## 0) Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```
**Key dependencies:** PyTorch, timm, albumentations, OpenCV, NumPy, matplotlib, PyYAML

---

## 1) Dataset Preparation

### 1.1 COCO2017 (for Object Detection & Pose Estimation)

Download COCO2017 using the helper script:

```shell script
./scripts/get_coco2017.sh
```

### 1.2 BIWI Dataset (for Head Rotation Estimation)

Download the dataset from [BIWI Database](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)

> **Note:** You can also download the dataset directly from the [Kaggle.](https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database)

---

## 2) Data Preprocessing

### 2.1 Single-Person Pose Estimation (MoveNet)

Prepares COCO keypoints by cropping individual person instances with sufficient visible keypoints:

```shell script
python scripts/make_coco2017_for_kpts.py \
  --root ./data/coco2017 \
  --out-dir ./data/coco2017_movenet_sp \
  --splits train,val \
  --min-visible-kpts 8 \
  --expand-ratio 1.4 \
  --jpeg-quality 95
```


**Output:**
```
data/coco2017_movenet_sp/
├── images/{train2017,val2017}/*.jpg
└── annotations/{person_keypoints_train2017.json, person_keypoints_val2017.json}
```


**Key parameters:**
- `--min-visible-kpts`: Minimum number of visible keypoints per instance
- `--expand-ratio`: Crop expansion factor around keypoints bounding box
- `--jpeg-quality`: Output image quality (1-100)

---

### 2.2 Object Detection (SSDLite)

Filters and subsets COCO detection annotations:

```shell script
# Example: Keep 5 classes
python scripts/make_coco2017_for_dets.py \
  --root ./data/coco2017 \
  --out-dir ./data/coco2017_det_5cls \
  --class-names person,car,bicycle,dog,chair \
  --skip-crowd \
  --min-box-area 16
```


**Options:**
- `--class-names`: Comma-separated list of classes to keep (omit for all 80 classes)
- `--skip-crowd`: Remove crowd annotations
- `--min-box-area`: Filter out tiny bounding boxes

> **Tip:** You can also keep all classes during preprocessing and filter at train time via `task_params.class_filter` in your config.

---

### 2.3 Head Rotation Estimation (6DRepNet)

The head rotation model uses the **BIWI Kinect Head Pose Database**. 

**Dataset structure expected:**
```
data/biwi/
├── 01/
│   ├── rgb.cal          # Camera intrinsics
│   ├── frame_00003_rgb.png
│   ├── frame_00003_pose.txt  # Rotation matrix
│   └── ...
├── 02/
└── ...
```


**Notes:**
- No preprocessing script needed – the dataset loader (`sparrow/datasets/biwi_rotation.py`) handles loading
- Download from: [BIWI Database](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)
- The loader reads rotation matrices from `*_pose.txt` files and converts them to 6D representation

---

## 3) Configuration

Each task has a YAML config file in `configs/`:

- **`configs/detection.yaml`** – Object detection (SSDLite)
- **`configs/keypoints.yaml`** – Pose estimation (MoveNet)
- **`configs/rotation.yaml`** – Head rotation estimation (SixDRepNet)

Refer to individual config files for task-specific parameters.

---

## 4) Training

Unified CLI for all three tasks:

```shell script
# Object Detection
python sparrow_cli.py train -t ssdlite -c configs/detection.yaml

# Pose Estimation
python sparrow_cli.py train -t movenet -c configs/keypoints.yaml

# Head Rotation
python sparrow_cli.py train -t sixrepnet -c configs/rotation.yaml
```


**Checkpoint management:**
- Set `resume: true` in config to auto-resume from `save_dir/last.pt`
- Training saves `best.pt` (lowest val loss) and `last.pt` (latest epoch)

---

## 5) Evaluation

Evaluate a trained model on the validation set:

```shell script
# Object Detection
python sparrow_cli.py eval -t ssdlite -c configs/detection.yaml --weights runs/detection/best.pt

# Pose Estimation
python sparrow_cli.py eval -t movenet -c configs/keypoints.yaml --weights runs/pose/best.pt

# Head Rotation
python sparrow_cli.py eval -t sixrepnet -c configs/rotation.yaml --weights runs/rotation/best.pt
```


**Output metrics:**
- **SSDLite**: mAP@0.5, mAP@0.5:0.95, precision, recall
- **MoveNet**: PCK@0.2 (keypoint accuracy), AUC
- **SixDRepNet**: MAE (yaw, pitch, roll), geodesic distance

---

## 6) Export

Export trained models to ONNX or TorchScript for deployment:

```shell script
# Export to ONNX
python sparrow_cli.py export -t ssdlite -c configs/detection.yaml 
```


**Default output:** `{save_dir}/export/model.onnx` or `model.pt`

---

## 7) Project Structure

```
MobileSparrow/
├── configs/              # YAML configs for each task
├── sparrow/
│   ├── models/          # Model definitions (FPN, SSDLite, MoveNet, SixDRepNet, etc.)
│   ├── datasets/        # Dataset loaders (COCO, BIWI, etc.)
│   ├── losses/          # Task-specific loss functions
│   └── trainer/         # Training/evaluation logic per task
├── scripts/             # Data preprocessing scripts
├── sparrow_cli.py       # Unified CLI entry point
└── requirements.txt
```


---

## 8) Model Architecture Overview

### Common Pattern
All models follow: **Backbone → FPN → Task Head**

- **Backbone**: Any `timm` model (MobileNetV3, EfficientNet, ShuffleNet, etc.)
- **FPN**: Feature Pyramid Network for multi-scale fusion
- **Heads**:
  - **SSDLite**: Multi-scale bounding box + class prediction
  - **MoveNet**: Heatmap + offset regression for keypoints
  - **SixDRepNet**: 6D rotation vector prediction for head pose

### Key Features
- **Lightweight**: Optimized for mobile/edge (depthwise separable convolutions)
- **Modular**: Swap backbones via config without code changes
- **Unified**: Single training/evaluation pipeline for all tasks

---

## Notes

- **Legacy configs**: Old configs using `num_classes: 17` for keypoints are auto-bridged, but new configs should use `task_params.num_joints`.
- **Data splits**: All loaders expect `train2017` and `val2017` subdirectories (no `test2017`).
- **Mixed precision**: Enable with `use_amp: true` in config (recommended for faster training).

---

## License

This project is licensed under the **GNU General Public License v3.0** (GPLv3).

See [LICENSE](LICENSE) for full details.

**Key points:**
- ✅ Free to use, modify, and distribute
- ✅ Commercial use allowed
- ⚠️ Must disclose source code of any modifications
- ⚠️ Derivative works must also be GPLv3

[![GPLv3](https://www.gnu.org/graphics/gplv3-88x31.png)](https://www.gnu.org/licenses/gpl-3.0.html)


## References

<sup>1</sup> MoveNet was originally proposed by **Ronny Votel and Na Li** from Google Research (2021) [[Blog]](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html). And `fire717` reimplemented it in [PyTorch](https://github.com/fire717/movenet.pytorch).
