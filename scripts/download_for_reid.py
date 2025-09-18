#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_reid.py
----------------
Download and organize ReID datasets (CUHK03, Market1501) into a unified folder:
  ./data/reid/
      images/                # unified JPG images
      final_pairs.csv        # img_path, person_id, image_id
      sources/               # (optional) per-dataset CSVs for traceability

Key features:
- Uses kagglehub to download:  priyanagda/cuhk03, rayiooo/reid_market-1501
- Converts all images to JPG (RGB), normalizes filenames -> {person_id}_{image_id}.jpg
- Remaps CUHK03 person IDs to start at 100000, Market1501 to 200000 (configurable)
- Reindexes image_id per person sequentially after merge
- Idempotent: skip steps if outputs exist unless --force

Usage:
  python prepare_reid.py --out ./data/reid
  python prepare_reid.py --out ./data/reid --no-cuhk03         # only Market1501
  python prepare_reid.py --out ./data/reid --no-market1501     # only CUHK03
  python prepare_reid.py --force                               # redo even if exists

Notes:
- You need internet + kagglehub installed (`pip install kagglehub pillow pandas`).
- If kagglehub cannot authenticate to Kaggle on your machine, configure Kaggle per docs.

"""

import os, sys, csv, argparse, shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from PIL import Image

# Soft deps; give clear errors if missing
def _require(pkgs: List[str]):
    missing = []
    for p in pkgs:
        try:
            __import__(p)
        except Exception:
            missing.append(p)
    if missing:
        raise RuntimeError(f"Missing packages: {missing}. Please install via: pip install {' '.join(missing)}")

_require(["pandas", "kagglehub", "PIL"])

import pandas as pd
import kagglehub


@dataclass
class Config:
    out_dir: Path
    cuhk03_start: int = 100000
    market_start: int = 200000
    use_cuhk03: bool = True
    use_market1501: bool = True
    force: bool = False
    jpg_quality: int = 95


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def convert_to_jpg(src: Path, dst: Path, quality: int = 95):
    """Convert src image to RGB JPEG at dst. Overwrites if exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(dst, format="JPEG", quality=quality, optimize=True)


def download_cuhk03() -> Path:
    print("[CUHK03] Downloading via kagglehub: priyanagda/cuhk03 ...")
    root = Path(kagglehub.dataset_download("priyanagda/cuhk03"))
    print(f"[CUHK03] Downloaded to: {root}")
    return root


def download_market1501() -> Path:
    print("[Market1501] Downloading via kagglehub: rayiooo/reid_market-1501 ...")
    root = Path(kagglehub.dataset_download("rayiooo/reid_market-1501"))
    print(f"[Market1501] Downloaded to: {root}")
    return root


def process_cuhk03(cuhk_root: Path, staging_dir: Path, pid_start: int) -> pd.DataFrame:
    """Return DataFrame with columns: img_path (relative to staging_dir), person_id, image_id"""
    labeled = cuhk_root / "archive" / "images_labeled"
    detected = cuhk_root / "archive" / "images_detected"
    if not labeled.exists():
        raise FileNotFoundError(f"[CUHK03] images_labeled folder not found at: {labeled}")

    rows = []
    out_dir = staging_dir / "images_cuhk03"
    ensure_dir(out_dir)

    # filenames like: 1_001_1_01.png
    for fn in sorted(labeled.glob("*.*")):
        name = fn.stem
        parts = name.split("_")
        if len(parts) < 4:
            # Skip unexpected
            print(f"[CUHK03] Skip unexpected filename: {fn.name}")
            continue
        try:
            # parts[1] is person id like '001'
            orig_pid = int(parts[1])
            person_id = pid_start + orig_pid
            # parts[3] is image index like '01'
            image_idx = int(parts[3])
        except Exception:
            print(f"[CUHK03] Skip unparsable filename: {fn.name}")
            continue

        # Tentative temp name before global reindex
        tmp_name = f"{person_id}_{image_idx:04d}.jpg"
        dst_rel = Path("images_cuhk03") / tmp_name
        dst = out_dir / tmp_name
        convert_to_jpg(fn, dst)

        rows.append({"img_path": str(dst_rel).replace("\\", "/"),
                     "person_id": person_id,
                     "image_id": image_idx})

    df = pd.DataFrame(rows)
    print(f"[CUHK03] Collected {len(df)} images from images_labeled.")
    return df


def process_market1501(market_root: Path, staging_dir: Path, pid_start: int) -> pd.DataFrame:
    """Return DataFrame with columns: img_path (relative to staging_dir), person_id, image_id"""
    bb_train = market_root / "bounding_box_train"
    if not bb_train.exists():
        # Some zips unpack into a nested folder; try to discover it
        candidates = list(market_root.glob("**/bounding_box_train"))
        if candidates:
            bb_train = candidates[0]
    if not bb_train.exists():
        raise FileNotFoundError(f"[Market1501] bounding_box_train not found under: {market_root}")

    rows = []
    out_dir = staging_dir / "images_market1501"
    ensure_dir(out_dir)

    # filenames like: 0002_c1s1_068496.png
    for fn in sorted(bb_train.glob("*.*")):
        stem = fn.stem
        parts = stem.split("_")
        if len(parts) < 3:
            print(f"[Market1501] Skip unexpected filename: {fn.name}")
            continue
        try:
            pid = int(parts[0])
            if pid < 0:
                # junk/distractor in test set; train set usually no -1, but just in case
                continue
            person_id = pid_start + pid
            image_idx = int(parts[2])
        except Exception:
            print(f"[Market1501] Skip unparsable filename: {fn.name}")
            continue

        tmp_name = f"{person_id}_{image_idx:06d}.jpg"
        dst_rel = Path("images_market1501") / tmp_name
        dst = out_dir / tmp_name
        convert_to_jpg(fn, dst)

        rows.append({"img_path": str(dst_rel).replace("\\", "/"),
                     "person_id": person_id,
                     "image_id": image_idx})

    df = pd.DataFrame(rows)
    print(f"[Market1501] Collected {len(df)} images from bounding_box_train.")
    return df


def merge_and_finalize(staging_dir: Path, out_dir: Path,
                       dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge per-dataset DataFrames, reindex image_id within each person, copy into out_dir/images,
       and write final_pairs.csv. Returns the final DataFrame.
    """
    ensure_dir(out_dir)
    images_dir = out_dir / "images"
    ensure_dir(images_dir)

    merged = pd.concat(dfs, ignore_index=True)
    # Reindex image_id per person sequentially
    merged["image_id"] = merged.groupby("person_id").cumcount() + 1

    # Create final relative path: images/{pid}_{image_id}.jpg
    merged["final_img_path"] = merged.apply(
        lambda r: f"images/{int(r['person_id'])}_{int(r['image_id']):04d}.jpg", axis=1
    )

    # Copy (convert already done) from staging into final
    n_copied = 0
    for idx, row in merged.iterrows():
        src = staging_dir / row["img_path"]
        dst = out_dir / row["final_img_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        n_copied += 1
        if n_copied % 1000 == 0:
            print(f"  Copied {n_copied} images...")

    # Prepare final CSV
    final_df = merged[["final_img_path", "person_id", "image_id"]].rename(
        columns={"final_img_path": "img_path"}
    )
    final_csv = out_dir / "final_pairs.csv"
    final_df.to_csv(final_csv, index=False)
    print(f"[FINAL] Wrote {len(final_df)} rows -> {final_csv}")

    # Also save per-dataset sources for traceability
    src_dir = out_dir / "sources"
    ensure_dir(src_dir)
    for i, df in enumerate(dfs):
        tag = "cuhk03" if "cuhk03" in df["img_path"].iloc[0] else "market1501"
        df.to_csv(src_dir / f"{tag}.csv", index=False)

    return final_df


def main():
    parser = argparse.ArgumentParser(description="Prepare ReID datasets into a unified folder")
    parser.add_argument("--out", type=str, default="./data/reid", help="Output root directory")
    parser.add_argument("--no-cuhk03", action="store_true", help="Skip CUHK03")
    parser.add_argument("--no-market1501", action="store_true", help="Skip Market1501")
    parser.add_argument("--cuhk03-start", type=int, default=100000, help="CUHK03 person_id offset")
    parser.add_argument("--market-start", type=int, default=200000, help="Market1501 person_id offset")
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs exist")
    args = parser.parse_args()

    cfg = Config(
        out_dir=Path(args.out).resolve(),
        cuhk03_start=args.cuhk03_start,
        market_start=args.market_start,
        use_cuhk03=not args.no_cuhk03,
        use_market1501=not args.no_market1501,
        force=args.force,
    )

    print("=== ReID Dataset Preparation ===")
    print(f"Output dir: {cfg.out_dir}")
    print(f"Use CUHK03: {cfg.use_cuhk03} (start={cfg.cuhk03_start})")
    print(f"Use Market1501: {cfg.use_market1501} (start={cfg.market_start})")

    final_csv = cfg.out_dir / "final_pairs.csv"
    if final_csv.exists() and not cfg.force:
        print(f"[SKIP] {final_csv} already exists. Use --force to rebuild.")
        return

    staging_dir = cfg.out_dir / "_staging"
    ensure_dir(staging_dir)

    dfs = []

    if cfg.use_cuhk03:
        cuhk_root = download_cuhk03()
        df_cuhk = process_cuhk03(cuhk_root, staging_dir, cfg.cuhk03_start)
        dfs.append(df_cuhk)
    if cfg.use_market1501:
        market_root = download_market1501()
        df_market = process_market1501(market_root, staging_dir, cfg.market_start)
        dfs.append(df_market)

    if not dfs:
        print("Nothing to process. Enable at least one dataset.")
        return

    merge_and_finalize(staging_dir, cfg.out_dir, dfs)

    # Clean up staging to save space
    try:
        shutil.rmtree(staging_dir)
        print(f"[CLEAN] Removed staging dir: {staging_dir}")
    except Exception as e:
        print(f"[WARN] Could not remove staging dir: {e}")

    print("=== DONE ===")


if __name__ == "__main__":
    main()
