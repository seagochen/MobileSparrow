# Movenet.Pytorch

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fire717/Fire/blob/main/LICENSE) 

## Intro
![start](/data/imgs/three_pane_aligned.gif)

MoveNet is an ultra fast and accurate model that detects 17 keypoints of a body.
This is A Pytorch implementation of MoveNet from Google. Include training code and pre-train model.

Google just release pre-train models(tfjs or tflite), which cannot be converted to some CPU inference framework such as NCNN,Tengine,MNN,TNN, and we can not add our own custom data to finetune, so there is this repo.


## How To Run

1. Download COCO dataset2017 from https://cocodataset.org/. Or Use the following script to automatically download the dataset into the right place.

```bash
./scripts/get_coco2017.sh
```

After the download is finished, cd to `data` directory and verify the data structure whether it is correct.

```bash
$ tree -L 2
.
├── coco2017
│   ├── annotations
│   ├── train2017
│   └── val2017
└── imgs
    ├── bad.png
    ├── good.png
    └── three_pane_aligned.gif
```

2. Use the following script to prepare the data:

```bash
# 1) By default, the original COCO split is used (train/val are exported separately)
python scripts/make_coco2017_for_movenet.py

# 2) After merging the COCO (that you specify), randomly re-split into train/test according to the ratio
python scripts/make_coco2017_for_movenet.py \
  --split-strategy random \
  --splits train,val \
  --train-ratio 0.9 \
  --seed 42 \
  --expand-ratio 1.25 \
  --min-visible-kpts 8 \
  --root ./data/coco2017 \
  --out-dir ./data/coco2017_movenet_sp
```

保留 COCO 原始划分，仅过滤 crowd、小框，且只保留 person 类（类名过滤）并用软链接省空间：

python make_coco2017_for_ssdlite.py \
  --root ./data/coco2017 \
  --out-dir ./data/coco2017_det_person \
  --split-strategy coco \
  --class-names person \
  --skip-crowd \
  --min-box-area 16 \
  --copy-mode symlink


做一个 5 类子集（person, car, bicycle, dog, chair），随机 9:1 重划分 train/test，并拷贝+重编码压缩图片：

python make_coco2017_for_ssdlite.py \
  --root ./data/coco2017 \
  --out-dir ./data/coco2017_det_5cls \
  --splits train,val \
  --split-strategy random --train-ratio 0.9 --seed 42 \
  --class-names person,car,bicycle,dog,chair \
  --skip-crowd --min-box-area 25 \
  --copy-mode copy --reencode --jpeg-quality 92


使用类ID过滤（例如只保留类别 id=1 和 3）：

python make_coco2017_for_ssdlite.py \
  --root ./data/coco2017 \
  --out-dir ./data/coco2017_det_cls13 \
  --class-ids 1,3 --skip-crowd --copy-mode symlink


3. You can add your own data to the same format.

4. Before training your own model, prepare the configuration json file first.

```json
{
  "GPU_ID": "0",
  "random_seed": 42,
  "cfg_verbose": true,

  "task": "kpts",                         // "kpts" | "det" | "cls"
  "task_params": {
    "num_joints": 17,                     // Only for 'kpts' task
    "export_keypoints": true,             // Only for 'kpts' task
    "class_agnostic_nms": false,          // Only for 'det' task
    "class_filter": [],                   // Only for 'det' task
    "cls_mode": "single_label"            // cls
  },

  "save_dir": "output/",
  "dataset_root_path": "./data/coco2017",

  "backbone": "mobilenet_v2",
  "width_mult": 1.0,
  "img_size": 256,
  "target_stride": 4,                     // Only for 'kpts' task

  "use_color_aug": true,
  "use_flip": true,
  "use_rotate": true,
  "rotate_deg": 30.0,
  "use_scale": true,
  "scale_range": [0.75, 1.25],
  "gaussian_radius": 2,                   // Only for 'kpts' task
  "sigma_scale": 1.0,                     // Only for 'kpts' task
  "select_person": "largest",             // Only for 'kpts' task

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

5. After choose 


## Training Results

#### Some good samples
![good](/data/imgs/good.png)

#### Some bad cases
![bad](/data/imgs/bad.png)


## Tips to improve
#### 1. Focus on data
* Add COCO2014. (But as I know it has some duplicate data of COCO2017, and I don't know if google use it.)
* Clean the croped COCO2017 data. (Some img just have little points, such as big face, big body,etc.MoveNet is a small network, COCO data is a little hard for it.)
* Add some yoga, fitness, and dance videos frame from YouTube. (Highly Recommened! Cause Google did this on their Movenet and said 'Evaluations on the Active validation dataset show a significant performance boost relative to identical architectures trained using only COCO. ')

#### 2. Change backbone
Try to ransfer Mobilenetv2(original Movenet) to Mobilenetv3 or Shufflenetv2 may get a litte improvement.If you just wanna reproduce the original Movenet, u can ignore this.

#### 3. More fancy loss
Surely this is a muti-task learning. So add some loss to learn together may improve the performence. (Such as BoneLoss which I have added.) And we can never know how Google trained, cause we cannot see it from the pre-train tflite model file, so you can try any loss function you like.


#### 4. Data Again
I just wanna you know the importance of the data. The more time you spend on clean data and add new data, the better performance your model will get! (While tips 2 and 3 may not.)

## Resource
1. [Blog:Next-Generation Pose Detection with MoveNet and TensorFlow.js](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html
)
2. [model card](https://storage.googleapis.com/movenet/MoveNet.SinglePose%20Model%20Card.pdf)
3. [TFHub：movenet/singlepose/lightning
](https://tfhub.dev/google/movenet/singlepose/lightning/4
)
4. [My article shared: 2021轻量级人体姿态估计模型修炼之路（附谷歌MoveNet复现经验）](https://zhuanlan.zhihu.com/p/413313925)



