# -*- coding: utf-8 -*-
"""
run_model_tests.py

为 MoveNet / OCRDetDB / OCRRecCTC / ReIDLite / SSDLite 提供：
1) 随机图像单次前向“能跑通”测试
2) CPU/GPU（如可用）各自重复 N 次的平均耗时
3) OCR det+rec 联合流程，从一张图得到最终的“类别 ID 序列”（未提供字典映射）

使用示例：
  python run_model_tests.py --which all --device all --backbone mobilenet_v3 --repeats 10 --warmup 3
  python run_model_tests.py --which ocr_pipeline --image /path/to/test.jpg --device cpu
  python run_model_tests.py --which ssdlite --device gpu --backbone shufflenet_v2 --outdir ./logs

参数：
  --which    all|movenet|ocr_det|ocr_rec|reid|ssdlite|ocr_pipeline
  --device   cpu|gpu|all
  --backbone mobilenet_v2|mobilenet_v3|shufflenet_v2
  --repeats  每个设备重复次数（默认 10）
  --warmup   预热次数（默认 3）
  --image    OCR 联合测试的输入图片路径（可选；缺省用随机图）
  --outdir   结果输出目录（默认当前目录）
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F


# ---------- 以脚本所在目录为项目根 ----------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------- 把本地 backbones 伪装成包以适配 import 路径 ----------
def ensure_backbone_pkg():
    """
    你的模型文件使用了类似：
      from sparrow.models.backbones.mobilenet_v3 import MobileNetV3Backbone
    这里自动把同目录下的 mobilenet_v2.py / mobilenet_v3.py / shufflenet_v2.py
    拷贝到 sparrow/models/backbones/，并写入 __init__.py，避免包导入失败。
    """
    pkg_root = ROOT / "sparrow" / "models" / "backbones"
    pkg_root.mkdir(parents=True, exist_ok=True)

    for p in [ROOT / "sparrow", ROOT / "sparrow" / "models", pkg_root]:
        init_file = p / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# auto-generated for tests\n", encoding="utf-8")

    for fname in ["mobilenet_v2.py", "mobilenet_v3.py", "shufflenet_v2.py"]:
        src = ROOT / fname
        dst = pkg_root / fname
        if src.exists():
            if not dst.exists() or src.read_bytes() != dst.read_bytes():
                dst.write_bytes(src.read_bytes())


ensure_backbone_pkg()


# ---------- 导入模型模块 ----------
from sparrow.models.movenet import MoveNet
from sparrow.models.ssdlite import SSDLite
from sparrow.models.reidlite import ReIDLite
from sparrow.models.ocr_rec import OCRRecCTC
from sparrow.models.ocr_det import OCRDetDB


# ---------- 工具函数 ----------
def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)


def timer_cpu(fn, warmup: int, repeats: int) -> float:
    """CPU 计时（ms）。"""
    for _ in range(max(warmup, 0)):
        fn()
    t0 = time.perf_counter()
    for _ in range(max(repeats, 1)):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / max(repeats, 1) * 1000.0


def timer_gpu(fn, warmup: int, repeats: int) -> float:
    """GPU 计时（ms），包含 cuda synchronize。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    torch.cuda.synchronize()
    for _ in range(max(warmup, 0)):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(max(repeats, 1)):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / max(repeats, 1) * 1000.0


def make_rand_img(size_hw: Tuple[int, int]) -> torch.Tensor:
    """生成随机张量 (1,3,H,W)。"""
    H, W = size_hw
    return torch.randn(1, 3, H, W)


def pil_load_image(path: str, size_hw: Tuple[int, int]) -> torch.Tensor:
    """读取图片并缩放到指定 HxW，失败则回退到随机图像。"""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("RGB").resize((size_hw[1], size_hw[0]))
        arr = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        return arr.unsqueeze(0)
    except Exception as e:
        print(f"[WARN] 读取图片失败：{e}；改用随机图像。")
        return make_rand_img(size_hw)


def save_results(outdir: Path, name: str, payload: Dict[str, Any]) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out_json = outdir / f"{name}_{ts}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] 结果已保存：{out_json}")
    return out_json


def shape_of(o):
    if isinstance(o, torch.Tensor):
        return list(o.shape)
    if isinstance(o, dict):
        return {k: shape_of(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [shape_of(v) for v in o]
    return str(type(o))


# ---------- 构建模型 ----------
def build_model(model_name: str, backbone: str, device: torch.device):
    if model_name == "movenet":
        model = MoveNet(backbone=backbone, width_mult=1.0).to(device).eval()
        input_size = (192, 192)
    elif model_name == "ocr_det":
        model = OCRDetDB(backbone=backbone, width_mult=1.0, use_fixed_thresh=True).to(device).eval()
        input_size = (256, 256)
    elif model_name == "ocr_rec":
        model = OCRRecCTC(backbone=backbone, width_mult=1.0, num_classes=96).to(device).eval()
        input_size = (32, 128)
    elif model_name == "reid":
        model = ReIDLite(backbone=backbone, width_mult=1.0).to(device).eval()
        input_size = (256, 128)
    elif model_name == "ssdlite":
        model = SSDLite(num_classes=80, backbone=backbone, width_mult=1.0).to(device).eval()
        input_size = (256, 256)
    else:
        raise ValueError(f"未知模型：{model_name}")
    return model, input_size


# ---------- 单模型测试（随机图像 + 计时） ----------
def test_single(model_name: str, backbone: str, device_str: str, repeats: int, warmup: int) -> Dict[str, Any]:
    report = {"model": model_name, "backbone": backbone, "device": device_str}
    device = torch.device(device_str)

    model, input_size = build_model(model_name, backbone, device)
    x = make_rand_img(input_size).to(device)

    with torch.no_grad():
        # 单次前向（验证能跑通）
        out = model(x)

        # 多次前向计时
        def _run():
            _ = model(x)

        if device.type == "cuda":
            avg_ms = timer_gpu(_run, warmup, repeats)
        else:
            avg_ms = timer_cpu(_run, warmup, repeats)

    report["input_size_hw"] = list(input_size)
    report["output_shape"] = shape_of(out)
    report["avg_ms"] = avg_ms
    return report


# ---------- OCR 联合流程：det -> 简单 bbox -> rec -> CTC 贪心解码 ----------
def extract_single_bbox(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    给一个二值 mask (1,H,W) 或 (H,W)，返回包含所有前景的外接矩形 (y0,x0,y1,x1)。
    若全为 0，则回退到中心 80% 区域。
    """
    if mask.dim() == 3:
        mask = mask[0]
    m = (mask > 0).nonzero(as_tuple=False)
    H, W = mask.shape[-2:]
    if m.numel() == 0:
        cy0, cx0 = int(0.1 * H), int(0.1 * W)
        cy1, cx1 = int(0.9 * H), int(0.9 * W)
        return cy0, cx0, cy1, cx1
    y0 = int(m[:, -2].min().item())
    y1 = int(m[:, -2].max().item()) + 1
    x0 = int(m[:, -1].min().item())
    x1 = int(m[:, -1].max().item()) + 1
    return y0, x0, y1, x1


def resize_rec_input(crop: torch.Tensor, h: int = 32, w: int = 128) -> torch.Tensor:
    """(B,3,H,W) -> (B,3,32,128)"""
    return F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)


def ocr_pipeline(backbone: str, device_str: str, repeats: int, warmup: int, image_path: str = None) -> Dict[str, Any]:
    device = torch.device(device_str)
    det_model = OCRDetDB(backbone=backbone, width_mult=1.0, use_fixed_thresh=True).to(device).eval()
    rec_model = OCRRecCTC(backbone=backbone, width_mult=1.0, num_classes=96).to(device).eval()

    # 准备输入（与 det 的输入尺寸保持一致）
    x = pil_load_image(image_path, (256, 256)) if image_path else make_rand_img((256, 256))
    x = to_device(x, device)

    with torch.no_grad():
        # --- 检测 ---
        det_out = det_model(x)                       # 期望含 'prob_map'
        prob = det_out["prob_map"]                   # (B,1,H,W) 或 (B,H,W)
        if prob.dim() == 4 and prob.size(1) == 1:
            pm = prob[:, 0]
        elif prob.dim() == 3:
            pm = prob
        else:
            raise RuntimeError(f"未知 prob_map 形状：{list(prob.shape)}")

        mask = (pm > 0.3).float()
        y0, x0, y1, x1 = extract_single_bbox(mask[0])

        # --- 识别 ---
        crop = x[:, :, y0:y1, x0:x1]
        rec_in = resize_rec_input(crop)              # (1,3,32,128)
        rec_out = rec_model(rec_in)
        logits = rec_out["logits"]                   # (B,T,C)
        ids = OCRRecCTC.ctc_greedy_decode(logits, blank_id=0)
        seq = ids[0] if len(ids) else []

        # 计时（det + rec 各自 repeats 次）
        if device.type == "cuda":
            t_det_ms = timer_gpu(lambda: det_model(x), warmup, repeats)
            t_rec_ms = timer_gpu(lambda: rec_model(rec_in), warmup, repeats)
        else:
            t_det_ms = timer_cpu(lambda: det_model(x), warmup, repeats)
            t_rec_ms = timer_cpu(lambda: rec_model(rec_in), warmup, repeats)

    return {
        "device": device_str,
        "backbone": backbone,
        "image_size": [int(x.shape[-2]), int(x.shape[-1])],
        "det": {
            "prob_map_shape": list(prob.shape),
            "bbox_yxxy": [y0, x0, y1, x1],
            "avg_ms": t_det_ms
        },
        "rec": {
            "logits_shape": list(logits.shape),
            "seq_ids": list(map(int, seq)),  # 如需转文字，请提供 charset 映射
            "avg_ms": t_rec_ms
        }
    }


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, default="all",
                        help="all|movenet|ocr_det|ocr_rec|reid|ssdlite|ocr_pipeline")
    parser.add_argument("--device", type=str, default="all", help="cpu|gpu|all")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3",
                        help="mobilenet_v2|mobilenet_v3|shufflenet_v2")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--image", type=str, default=None, help="OCR 联合测试的输入图片路径（可选）")
    parser.add_argument("--outdir", type=str, default=".", help="结果输出目录（默认当前目录）")
    args = parser.parse_args()

    tasks = ["movenet", "ocr_det", "ocr_rec", "reid", "ssdlite", "ocr_pipeline"]
    if args.which != "all":
        if args.which not in tasks:
            raise SystemExit(f"未知 --which={args.which}")
        tasks = [args.which]

    devices = []
    if args.device in ("cpu", "all"):
        devices.append("cpu")
    if args.device in ("gpu", "all") and torch.cuda.is_available():
        devices.append("cuda")
    elif args.device in ("gpu", "all") and not torch.cuda.is_available():
        print("[WARN] 请求 GPU，但当前环境 CUDA 不可用，自动跳过。")

    outdir = Path(args.outdir).resolve()

    results: List[Dict[str, Any]] = []
    for dev in devices:
        for t in tasks:
            try:
                if t == "ocr_pipeline":
                    payload = ocr_pipeline(args.backbone, dev, args.repeats, args.warmup, args.image)
                    results.append({"task": t, **payload})
                    print(f"[OK] {t} @ {dev}: det={payload['det']['avg_ms']:.3f}ms, rec={payload['rec']['avg_ms']:.3f}ms")
                else:
                    payload = test_single(t, args.backbone, dev, args.repeats, args.warmup)
                    results.append({"task": t, **payload})
                    print(f"[OK] {t} @ {dev}: avg={payload['avg_ms']:.3f}ms")
            except Exception as e:
                print(f"[ERR] {t} @ {dev}: {e}")
                results.append({"task": t, "device": dev, "backbone": args.backbone, "error": str(e)})

    save_results(outdir, "model_tests_results", {"args": vars(args), "results": results})


if __name__ == "__main__":
    main()
