#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO single-person square-crop generator (lean, keep official train/val only)
------------------------------------------------------------------------------
功能：
- 仅从 COCO 官方 person_keypoints_{train,val}2017.json 读取
- 仅保留“单人图”（同一张图中非 crowd 的 person 恰好 1 个）
- 过滤关键点数量不足的实例（--min-visible-kpts）
- 以 bbox ∪ 可见关键点 的最小外接矩形为基础做正方形裁剪（允许 padding）
- 写出与 COCO 兼容的 keypoints 标注（像素坐标，不归一化）
- 输出目录结构与现有 Loader 直接对接

输出结构：
<out_dir>/
  images/
    train2017/*.jpg
    val2017/*.jpg
  annotations/
    person_keypoints_train2017.json
    person_keypoints_val2017.json
"""
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from scripts import common


# ----------------------------
# 参数
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate single-person, square crops from COCO keypoints dataset (lean)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/coco2017"),
        help="COCO root dir. Must contain 'annotations', 'train2017', 'val2017'.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/coco2017_movenet_sp"),
        help="Output root directory.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Which COCO splits to read, comma in {train,val}.",
    )
    parser.add_argument(
        "--min-visible-kpts",
        type=int,
        default=8,
        help="Minimum visible (v>0) keypoints per instance.",
    )
    parser.add_argument(
        "--expand-ratio",
        type=float,
        default=1.0,
        help="Square crop side = max(w,h) of (bbox ∪ visible_kpts) * expand_ratio.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved crops.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra logs",
    )
    return parser.parse_args()


# ----------------------------
# 关键点/几何工具（与原脚本一致）
# ----------------------------
def visible_kpt_count(kpts: List[float]) -> int:
    # keypoints = [x1,y1,v1,x2,y2,v2,...] (len=51)
    return sum(1 for i in range(2, len(kpts), 3) if kpts[i] > 0)


def union_bbox_with_visible_kpts(bbox_xywh: List[float], kpts_xyv: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    x0, y0, x1, y1 = x, y, x + w, y + h
    for i in range(0, len(kpts_xyv), 3):
        xi, yi, vi = float(kpts_xyv[i]), float(kpts_xyv[i + 1]), float(kpts_xyv[i + 2])
        if vi > 0:
            x0 = min(x0, xi); y0 = min(y0, yi)
            x1 = max(x1, xi); y1 = max(y1, yi)
    return x0, y0, (x1 - x0), (y1 - y0)


def square_crop_from_box(box_xywh: Tuple[float, float, float, float], expand_ratio: float) -> Tuple[int, int, int]:
    bx, by, bw, bh = box_xywh
    cx, cy = bx + bw / 2.0, by + bh / 2.0
    side = int(math.ceil(max(bw, bh) * float(expand_ratio)))
    side = max(1, side)
    x0 = int(math.floor(cx - side / 2.0))
    y0 = int(math.floor(cy - side / 2.0))
    return x0, y0, side


def crop_with_padding(image: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1 = x0 + side, y0 + side
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(w, x1), min(h, y1)
    dst_x0, dst_y0 = max(0, -x0), max(0, -y0)
    dst_x1, dst_y1 = dst_x0 + (src_x1 - src_x0), dst_y0 + (src_y1 - src_y0)
    out = np.zeros((side, side, 3), dtype=image.dtype)
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def adjust_bbox_to_crop(bbox_xywh: List[float], crop_x0: int, crop_y0: int, side: int) -> List[float]:
    x, y, w, h = bbox_xywh
    nx, ny = float(x - crop_x0), float(y - crop_y0)
    nx = max(0.0, min(nx, float(side)))
    ny = max(0.0, min(ny, float(side)))
    return [nx, ny, float(w), float(h)]


def adjust_kpts_to_crop(kpts_xyv: List[float], crop_x0: int, crop_y0: int, side: int) -> List[float]:
    out: List[float] = []
    for i in range(0, len(kpts_xyv), 3):
        x, y, v = float(kpts_xyv[i]), float(kpts_xyv[i + 1]), float(kpts_xyv[i + 2])
        if v > 0:
            nx = float(min(side, max(0.0, x - crop_x0)))
            ny = float(min(side, max(0.0, y - crop_y0)))
            out.extend([nx, ny, v])
        else:
            out.extend([0.0, 0.0, 0.0])
    return out


# ----------------------------
# 数据结构
# ----------------------------
@dataclass
class ImgRec:
    split: str            # 'train' or 'val'
    file_name: str
    width: int
    height: int
    ann: Dict             # 唯一 person 标注
    crop_x0: int
    crop_y0: int
    crop_side: int


# ----------------------------
# 单图筛选 → 候选收集
# ----------------------------
def collect_single_person_candidates(root: Path, split: str, min_visible_kpts: int, expand_ratio: float, verbose: bool):
    img_dir = root / f"{split}2017"
    ann_path = root / "annotations" / f"person_keypoints_{split}2017.json"
    coco = common.load_json(ann_path)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]
    info = coco.get("info", {})
    licenses = coco.get("licenses", [])
    anns_by_img = common.index_by_image(annotations)

    # person 类 id
    person_cat_id = None
    for c in categories:
        if c.get("name") == "person":
            person_cat_id = c["id"]; break
    if person_cat_id is None:
        raise ValueError("No 'person' category found in categories.")

    kept: List[ImgRec] = []
    skipped_multi = 0
    skipped_kpt = 0

    for img in images:
        img_id = int(img["id"])
        file_name = img["file_name"]
        width = int(img["width"]); height = int(img["height"])

        all_anns = anns_by_img.get(img_id, [])
        person_anns = [a for a in all_anns if int(a.get("category_id")) == int(person_cat_id) and int(a.get("iscrowd", 0)) == 0]
        if len(person_anns) != 1:
            skipped_multi += 1
            continue

        ann = person_anns[0]
        kpts = ann.get("keypoints", [])
        if visible_kpt_count(kpts) < min_visible_kpts:
            skipped_kpt += 1
            continue

        bbox = ann["bbox"]
        u_x, u_y, u_w, u_h = union_bbox_with_visible_kpts(bbox, kpts)
        crop_x0, crop_y0, side = square_crop_from_box((u_x, u_y, u_w, u_h), expand_ratio)

        kept.append(ImgRec(
            split=split, file_name=file_name, width=width, height=height,
            ann=ann, crop_x0=crop_x0, crop_y0=crop_y0, crop_side=side
        ))

    if verbose:
        print(f"[collect:{split}] candidates={len(kept)} (skipped_multi={skipped_multi}, skipped_kpt={skipped_kpt})")
    return {
        "coco_info": dict(info=info, licenses=licenses, categories=categories),
        "img_dir": img_dir,
        "records": kept,
    }


# ----------------------------
# 导出工具（写 COCO JSON + 图像）
# ----------------------------
def save_crop_and_ann(rec: ImgRec, src_img_dir: Path, out_img_dir: Path,
                      next_image_id: int, next_ann_id: int, jpeg_quality: int):
    # 读图（兼容 unicode 路径）
    img_path = src_img_dir / rec.file_name
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None

    crop_img = crop_with_padding(img, rec.crop_x0, rec.crop_y0, rec.crop_side)

    # 映射 bbox & kpts（像素坐标）
    bbox = adjust_bbox_to_crop(rec.ann["bbox"], rec.crop_x0, rec.crop_y0, rec.crop_side)
    kpts = adjust_kpts_to_crop(rec.ann["keypoints"], rec.crop_x0, rec.crop_y0, rec.crop_side)
    num_kpts = visible_kpt_count(rec.ann["keypoints"])

    # 输出文件名（保持原名直观）
    base = Path(rec.file_name).stem
    out_name = f"{base}.jpg"
    out_path = out_img_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok, enc = cv2.imencode(".jpg", crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return None
    out_path.write_bytes(enc.tobytes())

    img_item = {
        "id": next_image_id,
        "file_name": out_name,
        "width": int(rec.crop_side),
        "height": int(rec.crop_side),
    }
    ann_item = {
        "id": next_ann_id,
        "image_id": next_image_id,
        "category_id": int(rec.ann["category_id"]),
        "iscrowd": 0,
        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
        "area": float(bbox[2] * bbox[3]),
        "segmentation": [],
        "num_keypoints": int(num_kpts),
        "keypoints": [float(v) for v in kpts],
    }
    return img_item, ann_item


# ----------------------------
# 主逻辑：保持官方 train/val
# ----------------------------
def main():
    args = parse_args()

    src_splits = [s.strip() for s in args.splits.split(",") if s.strip() in {"train", "val"}]
    if not src_splits:
        raise ValueError("No valid splits specified. Use --splits train,val or a subset.")
    
    common.validate_coco_root(args.root, src_splits, ann_prefix="person_keypoints")
    common.ensure_dir(args.out_dir / "images"); common.ensure_dir(args.out_dir / "annotations")

    # 逐个 split 导出
    ref_info = None; ref_licenses = None; ref_categories = None
    for sp in src_splits:
        bundle = collect_single_person_candidates(
            root=args.root, split=sp,
            min_visible_kpts=args.min_visible_kpts,
            expand_ratio=args.expand_ratio,
            verbose=args.verbose
        )
        if ref_info is None:
            ref_info = bundle["coco_info"]["info"]
            ref_licenses = bundle["coco_info"]["licenses"]
            ref_categories = bundle["coco_info"]["categories"]

        recs = bundle["records"]
        images, anns = [], []
        next_image_id, next_ann_id = 1, 1

        out_img_dir = args.out_dir / "images" / f"{sp}2017"
        out_ann_path = args.out_dir / "annotations" / f"person_keypoints_{sp}2017.json"

        for r in recs:
            result = save_crop_and_ann(
                rec=r,
                src_img_dir=bundle["img_dir"],
                out_img_dir=out_img_dir,
                next_image_id=next_image_id,
                next_ann_id=next_ann_id,
                jpeg_quality=args.jpeg_quality,
            )
            if result is None:
                continue
            img_item, ann_item = result
            images.append(img_item); anns.append(ann_item)
            next_image_id += 1; next_ann_id += 1

            if args.verbose and (next_image_id - 1) % 500 == 0:
                print(f"[{sp}] saved {next_image_id-1} items")

        common.write_coco_json(
            path=out_ann_path,
            info=ref_info,
            licenses=ref_licenses,
            categories=ref_categories,
            images=images,
            annotations=anns,
            desc_suffix=f"COCO 2017 {sp} single-person square crops (with padding), pixel coords",
        )
        print(f"[{sp}] Done. images={len(images)} anns={len(anns)} -> {out_ann_path}")


if __name__ == "__main__":
    main()
