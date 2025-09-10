#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO detection subset generator for SSDLite (lean)
--------------------------------------------------
从官方 COCO2017 生成 detection 训练集（instances_*），仅保留官方 train/val 划分：
- 通过 --splits 控制读入 train、val（默认 "train,val"）
- 类过滤（按 name 或 id）
- 过滤 iscrowd / 小框
- 图像复制（始终复制，绝不软链接/重编码）
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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

from scripts import common


# ----------------------------
# 参数
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare COCO detection subset for SSDLite (lean)."
    )
    p.add_argument(
        "--root", 
        type=Path, 
        default=Path("data/coco2017"),
        help="COCO root dir. Must contain 'annotations', and each requested split dir like 'train2017'/'val2017'."
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/coco2017_det"),
        help="Output root directory."
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Which COCO splits to read, comma in {train,val}."
    )

    # 类别过滤
    p.add_argument(
        "--class-names",
        type=str,
        default="",
        help="Comma-separated category names to keep (e.g., 'person,car,dog'). Empty=all."
    )
    p.add_argument(
        "--class-ids",
        type=str,
        default="",
        help="Comma-separated COCO category ids to keep (e.g., '1,3,18'). Empty=all."
    )
    p.add_argument(
        "--min-box-area",
        type=float,
        default=16.0,
        help="Filter tiny boxes by area (pixels^2) in *original image*."
    )
    p.add_argument(
        "--skip-crowd",
        action="store_true",
        help="Skip annotations with iscrowd=1."
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra logs."
    )
    return p.parse_args()


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
# 主流程：保持 COCO 原始 train/val
# ----------------------------
def run_coco_mode(root: Path, out_dir: Path,
                  splits: List[str],
                  keep_ids: Optional[Set[int]],
                  min_box_area: float, skip_crowd: bool,
                  verbose: bool):
    for split in splits:
        ann_path = root / "annotations" / f"instances_{split}2017.json"
        coco = common.load_json(ann_path)
        info = coco.get("info", {})
        licenses = coco.get("licenses", [])
        images, anns, categories = filter_and_remap_instances(
            coco, keep_ids, min_box_area, skip_crowd, verbose
        )

        # 落盘图像（始终复制）
        src_img_dir = root / f"{split}2017"
        dst_img_dir = out_dir / "images" / f"{split}2017"
        common.ensure_dir(dst_img_dir)
        kept_files = {im["file_name"] for im in images}
        for fn in kept_files:
            ok = common.copy_file(src_img_dir / fn, dst_img_dir / fn)
            if (not ok) and verbose:
                print(f"[warn] failed to copy: {src_img_dir / fn}")

        # 写 json
        out_ann = out_dir / "annotations" / f"instances_{split}2017.json"
        common.write_coco_json(out_ann, info, licenses, images, anns, categories,
                        desc_suffix=f"COCO 2017 {split} detection subset (filtered/contiguous ids).")
        print(f"[{split}] images={len(images)} anns={len(anns)} classes={len(categories)} -> {out_ann}")


# ----------------------------
# main
# ----------------------------
def main():
    args = parse_args()

    src_splits = [s.strip() for s in args.splits.split(",") if s.strip() in {"train", "val"}]
    if not src_splits:
        raise ValueError("No valid splits specified. Use --splits train,val or a subset.")

    common.validate_coco_root(args.root, src_splits)
    common.ensure_dir(args.out_dir / "images"); common.ensure_dir(args.out_dir / "annotations")

    # 类过滤（两者都为空 => 全部类别）
    # 从所选 splits 的第一个标注文件读取 categories，避免强制依赖 train 的 json
    cat_split = src_splits[0]
    coco_cats = common.load_json(args.root / "annotations" / f"instances_{cat_split}2017.json")["categories"]
    keep_ids = build_class_filter(coco_cats, args.class_names, args.class_ids)

    # 只保留官方划分
    run_coco_mode(
        root=args.root, out_dir=args.out_dir, splits=src_splits,
        keep_ids=keep_ids, min_box_area=args.min_box_area, skip_crowd=args.skip_crowd,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
