#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO detection subset generator for SSDLite (lean)
--------------------------------------------------
从官方 COCO2017 生成 detection 训练集（instances_*），仅保留官方 train/val 划分：
- 通过 --splits 控制读入 train、val（默认 "train,val"）
- 类过滤（按 name 或 id）
- 过滤 iscrowd / 小框
- 图像拷贝 / 软链接 / 不复制
输出结构：
<out_dir>/
  images/
    train2017/*.jpg
    val2017/*.jpg
  annotations/
    instances_train2017.json
    instances_val2017.json
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import cv2
import shutil
import os


# ----------------------------
# 参数
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Prepare COCO detection subset for SSDLite (lean).")
    p.add_argument("--root", type=Path, default=Path("data/coco2017"),
                   help="COCO root dir. Must contain 'annotations', 'train2017', 'val2017'.")
    p.add_argument("--out-dir", type=Path, default=Path("data/coco2017_det"),
                   help="Output root directory.")
    p.add_argument("--splits", type=str, default="train,val",
                   help="Which COCO splits to read, comma in {train,val}.")
    # 类过滤与清洗
    p.add_argument("--class-names", type=str, default="",
                   help="Comma-separated category names to keep (e.g., 'person,car,dog'). Empty=all.")
    p.add_argument("--class-ids", type=str, default="",
                   help="Comma-separated COCO category ids to keep (e.g., '1,3,18'). Empty=all.")
    p.add_argument("--min-box-area", type=float, default=16.0,
                   help="Filter tiny boxes by area (pixels^2) in *original image*.")
    p.add_argument("--skip-crowd", action="store_true", help="Skip annotations with iscrowd=1.")
    # 图像处理
    p.add_argument("--copy-mode", type=str, choices=["copy", "symlink", "none"], default="symlink",
                   help="'copy'=复制图片; 'symlink'=创建软链接(省空间); 'none'=不写图片（仅写JSON，训练需指向原COCO路径）")
    p.add_argument("--jpeg-quality", type=int, default=95,
                   help="If re-encoding when copying, set JPEG quality.")
    p.add_argument("--reencode", action="store_true",
                   help="When copying, re-encode to JPEG for smaller size. Otherwise do raw file copy.")
    p.add_argument("--verbose", action="store_true", help="Print extra logs.")
    return p.parse_args()


# ----------------------------
# 校验/加载
# ----------------------------
def validate_coco_root(root: Path) -> None:
    ann_dir = root / "annotations"
    train_dir = root / "train2017"
    val_dir = root / "val2017"
    for d in (ann_dir, train_dir, val_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Required directory not found: {d}")
    required_ann_files = [
        ann_dir / "instances_train2017.json",
        ann_dir / "instances_val2017.json",
    ]
    missing = [str(p) for p in required_ann_files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing annotation files:\n  " + "\n  ".join(missing))


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# IO 工具
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, mode: str, jpeg_quality: int = 95, reencode: bool = False) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "none":
        return True
    if mode == "symlink":
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
            return True
        except OSError:
            mode = "copy"  # Windows 等环境回退
    if mode == "copy":
        if reencode:
            img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return False
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if not ok:
                return False
            dst.write_bytes(enc.tobytes())
        else:
            shutil.copy2(src, dst)
        return True
    return False


# ----------------------------
# 读取类过滤
# ----------------------------
def build_class_filter(coco_cats: List[Dict], names_csv: str, ids_csv: str) -> Optional[Set[int]]:
    if not names_csv and not ids_csv:
        return None
    keep_ids: Set[int] = set()
    if ids_csv.strip():
        for s in ids_csv.split(","):
            s = s.strip()
            if s:
                keep_ids.add(int(s))
    if names_csv.strip():
        name2id = {c["name"]: int(c["id"]) for c in coco_cats}
        for s in names_csv.split(","):
            nm = s.strip()
            if nm:
                cid = name2id.get(nm)
                if cid is None:
                    raise ValueError(f"Category name not found in COCO: '{nm}'")
                keep_ids.add(int(cid))
    return keep_ids


# ----------------------------
# 过滤 + 重映射
# ----------------------------
def filter_and_remap_instances(coco: Dict,
                               keep_ids: Optional[Set[int]],
                               min_box_area: float,
                               skip_crowd: bool,
                               verbose: bool) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    返回: images, annotations, categories（只保留用到的类，且重建连续ID 1..C，背景0留给模型）
    """
    images = coco["images"]
    anns = coco["annotations"]

    used_img: Set[int] = set()
    kept_anns: List[Dict] = []
    for a in anns:
        cid = int(a["category_id"])
        if keep_ids and cid not in keep_ids:
            continue
        if skip_crowd and int(a.get("iscrowd", 0)) != 0:
            continue
        x, y, w, h = [float(v) for v in a["bbox"]]
        if w <= 0 or h <= 0 or (w * h) < float(min_box_area):
            continue
        kept_anns.append({
            "id": int(a["id"]),
            "image_id": int(a["image_id"]),
            "category_id": cid,          # 暂用旧 cid，下面统一重映射
            "iscrowd": 0,
            "bbox": [x, y, w, h],
            "area": float(w * h),
            "segmentation": [],
        })
        used_img.add(int(a["image_id"]))

    # 只保留包含有效目标的图片
    images = [im for im in images if int(im["id"]) in used_img]

    # 按用到的类重建 categories，并分配连续ID 1..C
    used_cids = sorted({int(a["category_id"]) for a in kept_anns})
    cid2new = {cid: i + 1 for i, cid in enumerate(used_cids)}   # 1..C
    cid2name = {int(c["id"]): c["name"] for c in coco["categories"]}
    categories = [{"id": cid2new[old], "name": cid2name.get(old, str(old))} for old in used_cids]

    # 重映射 annotations.category_id
    for a in kept_anns:
        a["category_id"] = cid2new[int(a["category_id"])]

    if verbose:
        print(f"[filter] images={len(images)} anns={len(kept_anns)} classes={len(categories)}")

    return images, kept_anns, categories


# ----------------------------
# 写 COCO JSON
# ----------------------------
def write_coco_json(path: Path, info: Dict, licenses: List[Dict], images: List[Dict],
                    annotations: List[Dict], categories: List[Dict], desc_suffix: str):
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
# 主流程：保持 COCO 原始 train/val
# ----------------------------
def run_coco_mode(root: Path, out_dir: Path,
                  keep_ids: Optional[Set[int]],
                  min_box_area: float, skip_crowd: bool,
                  copy_mode: str, reencode: bool, jpeg_quality: int,
                  verbose: bool):
    for split in ("train", "val"):
        ann_path = root / "annotations" / f"instances_{split}2017.json"
        coco = load_json(ann_path)
        info = coco.get("info", {})
        licenses = coco.get("licenses", [])
        images, anns, categories = filter_and_remap_instances(
            coco, keep_ids, min_box_area, skip_crowd, verbose
        )

        # 落盘/链接图像
        src_img_dir = root / f"{split}2017"
        dst_img_dir = out_dir / "images" / f"{split}2017"
        ensure_dir(dst_img_dir)
        kept_files = {im["file_name"] for im in images}
        for fn in kept_files:
            copy_or_link(src_img_dir / fn, dst_img_dir / fn, mode=copy_mode,
                         jpeg_quality=jpeg_quality, reencode=reencode)

        # 写 json
        out_ann = out_dir / "annotations" / f"instances_{split}2017.json"
        write_coco_json(out_ann, info, licenses, images, anns, categories,
                        desc_suffix=f"COCO 2017 {split} detection subset (filtered/contiguous ids).")
        print(f"[{split}] images={len(images)} anns={len(anns)} classes={len(categories)} -> {out_ann}")


# ----------------------------
# main
# ----------------------------
def main():
    args = parse_args()
    validate_coco_root(args.root)
    ensure_dir(args.out_dir / "images"); ensure_dir(args.out_dir / "annotations")

    src_splits = [s.strip() for s in args.splits.split(",") if s.strip() in {"train", "val"}]
    if not src_splits:
        raise ValueError("No valid splits specified. Use --splits train,val or a subset.")

    # 类过滤（两者都为空 => 全部类别）
    coco_cats = load_json(args.root / "annotations" / "instances_train2017.json")["categories"]
    keep_ids = build_class_filter(coco_cats, args.class_names, args.class_ids)

    # 只保留官方划分
    run_coco_mode(
        root=args.root, out_dir=args.out_dir,
        keep_ids=keep_ids, min_box_area=args.min_box_area, skip_crowd=args.skip_crowd,
        copy_mode=args.copy_mode, reencode=args.reencode, jpeg_quality=args.jpeg_quality,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
