# scripts/conv_utils.py
import json
import shutil
from pathlib import Path
from typing import Dict, List

# ----------------------------
# IO 工具
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path) -> bool:
    """始终复制原文件（不重编码）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False

# ----------------------------
# JSON / 读写
# ----------------------------
def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_coco_json(path: Path, info: Dict, licenses: List[Dict],
                    images: List[Dict], annotations: List[Dict],
                    categories: List[Dict], desc_suffix: str):
    out = {
        "info": {**info, "description": desc_suffix},
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

# ----------------------------
# 校验 / 索引
# ----------------------------
def validate_coco_root(root: Path, splits: List[str], ann_prefix: str) -> None:
    """校验 COCO 根目录结构 & 指定标注前缀（如 'instances' / 'person_keypoints'）。"""
    ann_dir = root / "annotations"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Required directory not found: {ann_dir}")
    for split in splits:
        img_dir = root / f"{split}2017"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Required directory not found: {img_dir}")
        ann_path = ann_dir / f"{ann_prefix}_{split}2017.json"
        if not ann_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")

def index_by_image(annotations: List[Dict]) -> Dict[int, List[Dict]]:
    by_img: Dict[int, List[Dict]] = {}
    for ann in annotations:
        by_img.setdefault(int(ann["image_id"]), []).append(ann)
    return by_img
