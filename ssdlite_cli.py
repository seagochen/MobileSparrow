#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSDLite Unified CLI
- train:        训练并保存 last.pt / best.pt
- eval:         在验证集上评估（loss proxy）
- predict:      对目录图片推理并可视化
- export-onnx:  导出 ONNX（三输出：cls_logits / bbox_regs / anchors）
- show-config:  打印合并后的最终配置

依赖：PyTorch、opencv-python、numpy、onnx(导出时)、onnxruntime(可选验证)
"""

import os
import cv2
import json
import glob
import math
import random
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader

import core
from core.datasets.coco_loader import CoCo2017DataLoader   # 统一入口（kpts/det/cls）  ← 使用它拿到 det 的 DataLoader
from core.models.ssdlite import SSDLiteDet                 # 你的 SSDLite 模型                    :contentReference[oaicite:5]{index=5}
from core.loss.ssd_loss import SSDLoss                     # 你的检测损失
from core.datasets.common import letterbox                 # 复用 letterbox 与训练对齐
# 参考：movenet_cli.py 的结构/参数覆盖/日志风格                               :contentReference[oaicite:6]{index=6}


# -------------------------
# utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_cfg(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(base)
    for k, v in override.items():
        if v is not None:
            cfg[k] = v
    return cfg

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


# -------------------------
# build model / data
# -------------------------
def build_data(cfg: Dict[str, Any]):
    """
    使用统一 DataLoader（task='det'）拿到训练/验证 Loader；
    同时 data.num_classes 会被填充为 背景+N，有助于自动构建模型。
    """
    data = CoCo2017DataLoader(cfg, task="det")  # ← 走你的新封装                    :contentReference[oaicite:7]{index=7}
    train_loader, val_loader = data.getTrainValDataloader()
    num_classes = data.num_classes if data.num_classes is not None else int(cfg.get("num_classes", 81))
    return train_loader, val_loader, num_classes

def build_model(cfg: Dict[str, Any], num_classes: int) -> torch.nn.Module:
    model = SSDLiteDet(
        backbone=cfg.get("backbone", "mobilenet_v2"),
        num_classes=num_classes,                           # 背景+N
        width_mult=float(cfg.get("width_mult", 1.0)),
        neck_outc=int(cfg.get("neck_outc", 64)),
        head_midc=int(cfg.get("head_midc", 64)),
        score_thresh=float(cfg.get("score_thresh", 0.35)),
        nms_thresh=float(cfg.get("nms_thresh", 0.5)),
        topk=int(cfg.get("topk", 200)),
        class_agnostic_nms=bool(cfg.get("class_agnostic_nms", False)),
    )
    return model


# -------------------------
# train / eval
# -------------------------
def cmd_train(cfg: Dict[str, Any]):
    core.init(cfg)
    set_seed(cfg.get("random_seed", 42))
    ensure_dir(cfg["save_dir"])

    device = torch.device("cuda" if (cfg.get("GPU_ID", "") != "" and torch.cuda.is_available()) else "cpu")

    train_loader, val_loader, num_classes = build_data(cfg)
    model = build_model(cfg, num_classes).to(device)
    criterion = SSDLoss(num_classes=num_classes, alpha=float(cfg.get("ssd_alpha", 1.0)))

    # optimizer / scheduler（与现有 DetTask 保持一致的风格）                    :contentReference[oaicite:8]{index=8}
    opt_name = cfg.get("optimizer", "Adam")
    if opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 3.5e-4)),
                                     weight_decay=float(cfg.get("weight_decay", 1e-4)))
    elif opt_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)),
                                    momentum=0.9, weight_decay=float(cfg.get("weight_decay", 1e-4)))
    else:
        raise ValueError(opt_name)

    sch_name = cfg.get("scheduler", "MultiStepLR-90,130-0.2")
    if "MultiStepLR" in sch_name:
        milestones = [int(x) for x in sch_name.strip().split('-')[1].split(',')]
        gamma = float(sch_name.strip().split('-')[2])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif "step" in sch_name:
        step_size = int(sch_name.strip().split('-')[1]); gamma = float(sch_name.strip().split('-')[2])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                               patience=5, min_lr=1e-6)

    best_proxy = -1e9
    epochs = int(cfg.get("epochs", 100))
    log_interval = int(cfg.get("log_interval", 20))
    grad_clip = float(cfg.get("clip_gradient", 0.0))

    for ep in range(epochs):
        model.train()
        run_loss = 0.0

        for it, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            t_boxes = [t["boxes"].to(device) for t in targets]
            t_labels= [t["labels"].to(device) for t in targets]
            batch_targets = {"boxes": t_boxes, "labels": t_labels}

            out = model(imgs)  # {"cls_logits","bbox_regs","anchors"}
            loss, meter = criterion(out["cls_logits"], out["bbox_regs"], out["anchors"], batch_targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            run_loss += float(loss.item())
            if it % log_interval == 0:
                print(f"\r[Train] ep {ep+1}/{epochs} it {it}/{len(train_loader)} "
                      f"loss {loss.item():.4f} (cls {meter['loss_cls']:.3f} reg {meter['loss_reg']:.3f}) "
                      f"pos {meter['pos']}", end='', flush=True)
        print()

        # 简化版验证：用 val loss 作为 proxy（越小越好 -> 记 best 的负数）
        val_loss = _validate(model, val_loader, criterion, device)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-val_loss)
        else:
            scheduler.step()

        # 保存
        last = os.path.join(cfg["save_dir"], "last.pt")
        torch.save(model.state_dict(), last)
        if -val_loss > best_proxy:
            best_proxy = -val_loss
            best = os.path.join(cfg["save_dir"], "best.pt")
            torch.save(model.state_dict(), best)
            print(f"[INFO] new best (proxy) -> {best}")

    print("[OK] Training finished.")

@torch.no_grad()
def _validate(model, val_loader, criterion, device):
    model.eval()
    tot, n = 0.0, 0
    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        t_boxes = [t["boxes"].to(device) for t in targets]
        t_labels= [t["labels"].to(device) for t in targets]
        batch_targets = {"boxes": t_boxes, "labels": t_labels}

        out = model(imgs)
        loss, meter = criterion(out["cls_logits"], out["bbox_regs"], out["anchors"], batch_targets)
        tot += float(loss.item()); n += 1
    avg = tot / max(1, n)
    print(f"[Val] loss {avg:.4f}")
    return avg


# -------------------------
# predict (folder)
# -------------------------
@torch.no_grad()
def cmd_predict(cfg: Dict[str, Any], images_dir: str, out_dir: str, weights: str | None):
    core.init(cfg)
    ensure_dir(out_dir)

    device = torch.device("cuda" if (cfg.get("GPU_ID", "") != "" and torch.cuda.is_available()) else "cpu")

    # 用验证集的类别数构建模型（或 fallback）
    _, val_loader, num_classes = build_data(cfg)
    model = build_model(cfg, num_classes).to(device).eval()

    # 自动找权重
    if not weights:
        for cand in (Path(cfg["save_dir"]) / "best.pt", Path(cfg["save_dir"]) / "last.pt"):
            if cand.exists():
                weights = str(cand); break
    if not weights:
        raise FileNotFoundError("未找到权重，请用 --weights 指定，或先训练得到 best.pt/last.pt。")
    model.load_state_dict(torch.load(weights, map_location=device))
    print(f"[INFO] loaded weights: {weights}")

    img_size = int(cfg.get("img_size", 256))

    # 遍历目录
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    files = sum([glob.glob(os.path.join(images_dir, e)) for e in exts], [])
    if not files:
        raise FileNotFoundError(f"未在 {images_dir} 找到图片")

    for p in files:
        bgr = cv2.imread(p); 
        if bgr is None: 
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # letterbox 与训练对齐
        img_lb, scale, (pad_w, pad_h) = letterbox(rgb, img_size, color=(114,114,114))
        x = torch.from_numpy(img_lb.transpose(2,0,1)).float().unsqueeze(0) / 255.0
        x = x.to(device)

        outs = model(x)  # 推理：List[dict] or 已拼接后处理（见你的实现）        :contentReference[oaicite:9]{index=9}
        # 兼容两种返回：训练 dict / 推理 list
        if isinstance(outs, dict):
            # 若模型在 eval 仍返回 dict，这里给一个极简的“最高类概率 + 解码 + NMS”的兜底
            boxes, scores, labels = _quick_decode_for_vis(outs, img_size, img_size)
            det = {"boxes": boxes, "scores": scores, "labels": labels}
        else:
            det = outs[0]

        vis = _draw_dets(img_lb.copy(), det)
        outp = os.path.join(out_dir, Path(p).stem + "_det.jpg")
        cv2.imwrite(outp, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"[OK] saved: {outp}")

def _quick_decode_for_vis(out: Dict[str, torch.Tensor], W: int, H: int, score_th: float = 0.35):
    # 与 SSDLiteDet._decode 一致的方差；仅作可视化兜底
    var = torch.tensor([0.1, 0.1, 0.2, 0.2], device=out["bbox_regs"].device)
    cls = torch.softmax(out["cls_logits"], dim=-1)[0]         # [N,C]
    loc = out["bbox_regs"][0]                                 # [N,4]
    anc = out["anchors"]                                      # [N,4] (cx,cy,w,h) 归一化

    # 解码到像素坐标（xyxy）
    dx,dy,dw,dh = loc.unbind(dim=1)
    acx,acy,aw,ah = anc.unbind(dim=1)
    px = dx * var[0] * aw + acx; py = dy * var[1] * ah + acy
    pw = torch.exp(dw * var[2]) * aw; ph = torch.exp(dh * var[3]) * ah
    x1 = (px - pw*0.5) * W; y1 = (py - ph*0.5) * H
    x2 = (px + pw*0.5) * W; y2 = (py + ph*0.5) * H
    boxes = torch.stack([x1,y1,x2,y2], dim=1).clamp_(min=0)

    scores, labels = cls.max(dim=1)                           # background 仍在内
    keep = scores > score_th
    return boxes[keep].detach().cpu().numpy(), scores[keep].detach().cpu().numpy(), labels[keep].detach().cpu().numpy()

def _draw_dets(img_rgb: np.ndarray, det: Dict[str, np.ndarray]):
    boxes = det["boxes"]; scores = det["scores"]; labels = det["labels"]
    im = img_rgb.copy()
    for (x1,y1,x2,y2), s, c in zip(boxes, scores, labels):
        x1,y1,x2,y2 = [int(v) for v in [x1,y1,x2,y2]]
        cv2.rectangle(im, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(im, f"{int(c)}:{s:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    return im


# -------------------------
# export onnx
# -------------------------
@torch.no_grad()
def cmd_export_onnx(cfg: Dict[str, Any], weights: str | None, out_path: str, opset: int, dynamic: bool, verify: bool):
    """
    导出三输出：
      - cls_logits: [B, N, C]
      - bbox_regs : [B, N, 4]
      - anchors   : [N, 4]  (cx,cy,w,h in [0,1])
    好处：跨平台后处理（NMS/解码）自由，避免 ONNX 算子不统一。
    """
    core.init(cfg)
    set_seed(cfg.get("random_seed", 42))
    ensure_dir(Path(out_path).parent)

    device = torch.device("cuda" if (cfg.get("GPU_ID", "") != "" and torch.cuda.is_available()) else "cpu")

    # 用数据集决定 num_classes
    _, _, num_classes = build_data(cfg)
    model = build_model(cfg, num_classes).to(device).eval()

    # 选择权重
    if not weights:
        for cand in (Path(cfg["save_dir"]) / "best.pt", Path(cfg["save_dir"]) / "last.pt"):
            if cand.exists():
                weights = str(cand); break
    if not weights:
        raise FileNotFoundError("未找到可用权重，请通过 --weights 指定，或先训练得到 best.pt/last.pt。")
    model.load_state_dict(torch.load(weights, map_location=device))

    # dummy
    h = w = int(cfg.get("img_size", 256))
    dummy = torch.randn(1, 3, h, w, device=device)

    class Wrapper(torch.nn.Module):
        def __init__(self, m: torch.nn.Module):
            super().__init__(); self.m = m
        def forward(self, x):
            y = self.m(x)  # 训练/评估均返回 dict（若评估返回 list，可改用内部路径拿 dict）
            if isinstance(y, dict):
                return y["cls_logits"], y["bbox_regs"], y["anchors"]
            else:
                # 若 eval 返回 list，则强制走训练路径
                self.m.train(False)
                z = self.m(x)
                return z["cls_logits"], z["bbox_regs"], z["anchors"]

    wrapper = Wrapper(model).to(device).eval()

    out_names = ["cls_logits", "bbox_regs", "anchors"]
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input":      {0: "batch", 2: "height", 3: "width"},
            "cls_logits": {0: "batch", 1: "num_priors"},
            "bbox_regs":  {0: "batch", 1: "num_priors"},
            "anchors":    {0: "num_priors"},
        }

    torch.onnx.export(
        wrapper, dummy, out_path,
        input_names=["input"], output_names=out_names,
        opset_version=opset, do_constant_folding=True,
        dynamic_axes=dynamic_axes, verbose=False
    )
    print(f"[OK] ONNX saved to: {out_path}")

    if verify:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(out_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
            outs = sess.run(None, {"input": dummy.detach().cpu().numpy()})
            assert len(outs) == 3
            print("[OK] ONNXRuntime quick check passed.")
        except Exception as e:
            print(f"[WARN] onnxruntime 验证失败（可忽略）：{e}")


# -------------------------
# CLI
# -------------------------
def build_parser():
    epilog_msg = """
示例:
  # 使用默认配置训练
  python ssdlite_cli.py --config config.json train --epochs 120 --batch-size 64

  # 在验证集上评估
  python ssdlite_cli.py --config config.json eval --weights output/best.pt

  # 目录预测并可视化
  python ssdlite_cli.py --config config.json predict --images ./test_imgs --out ./vis_det

  # 导出 ONNX（三输出）
  python ssdlite_cli.py --config config.json export-onnx --out output/ssdlite.onnx

  # 打印合并后的配置
  python ssdlite_cli.py --config config.json show-config
"""
    p = argparse.ArgumentParser(
        description="SSDLite Unified CLI (train/eval/predict/export/show-config)",
        epilog=epilog_msg,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 通用覆盖
    p.add_argument("--config", type=str, default="config.json", help="JSON 配置")
    p.add_argument("--save-dir", type=str, help="覆盖 cfg.save_dir")
    p.add_argument("--img-size", type=int, help="覆盖 cfg.img_size")
    p.add_argument("--num-classes", type=int, help="覆盖 cfg.num_classes（如未从数据自动获取时）")
    p.add_argument("--width-mult", type=float, help="覆盖 cfg.width_mult")
    p.add_argument("--backbone", type=str, help="覆盖 cfg.backbone (mobilenet_v2 / shufflenet_v2)")
    p.add_argument("--gpu-id", type=str, help="覆盖 cfg.GPU_ID (''=CPU, '0'=第0张GPU)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    sp_tr = sub.add_parser("train", help="训练并保存权重 (last.pt / best.pt)")
    sp_tr.add_argument("--epochs", type=int, help="训练轮数")
    sp_tr.add_argument("--batch-size", type=int, help="批大小")
    sp_tr.add_argument("--lr", type=float, help="学习率")
    sp_tr.add_argument("--optimizer", type=str, help="优化器 (Adam / SGD)")
    sp_tr.add_argument("--scheduler", type=str, help="学习率调度器, 如 'MultiStepLR-90,130-0.2'")

    # eval
    sp_ev = sub.add_parser("eval", help="在验证集上评估模型（loss proxy）")
    sp_ev.add_argument("--weights", type=str, help="权重路径 (默认 save_dir/best.pt)")

    # predict
    sp_pr = sub.add_parser("predict", help="对目录中的图片进行预测并可视化")
    sp_pr.add_argument("--images", type=str, required=True, help="输入图片目录")
    sp_pr.add_argument("--out", type=str, required=True, help="输出目录 (保存可视化结果)")
    sp_pr.add_argument("--weights", type=str, help="权重路径 (默认 save_dir/best.pt 或 last.pt)")

    # export-onnx
    sp_ex = sub.add_parser("export-onnx", help="导出 ONNX（cls_logits / bbox_regs / anchors）")
    sp_ex.add_argument("--weights", type=str, help="权重路径 (默认 best.pt / last.pt)")
    sp_ex.add_argument("--out", type=str, default="output/ssdlite.onnx", help="导出 ONNX 文件路径")
    sp_ex.add_argument("--opset", type=int, default=13, help="ONNX opset 版本 (默认 13)")
    sp_ex.add_argument("--dynamic", action="store_true", help="导出动态 batch/height/width 和先验数")
    sp_ex.add_argument("--verify", action="store_true", help="导出后用 onnxruntime 进行推理校验")

    # show-config
    sub.add_parser("show-config", help="打印最终合并后的配置 (JSON 格式)")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 载入 JSON 配置
    if not Path(args.config).exists():
        raise FileNotFoundError(f"未找到配置文件：{args.config}")
    base_cfg = load_json(args.config)

    # 命令行覆盖项（与 movenet_cli 对齐）                               # :contentReference[oaicite:10]{index=10}
    override = {
        "save_dir": args.save_dir,
        "img_size": args.img_size,
        "num_classes": args.num_classes,
        "width_mult": args.width_mult,
        "backbone": args.backbone,
        "GPU_ID": args.gpu_id,
        # 训练覆盖
        "epochs": getattr(args, "epochs", None),
        "batch_size": getattr(args, "batch_size", None),
        "learning_rate": getattr(args, "lr", None),
        "optimizer": getattr(args, "optimizer", None),
        "scheduler": getattr(args, "scheduler", None),
    }
    cfg = merge_cfg(base_cfg, override)

    if args.cmd == "show-config":
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return

    if args.cmd == "train":
        cmd_train(cfg)
    elif args.cmd == "eval":
        # 直接复用验证函数
        core.init(cfg)
        set_seed(cfg.get("random_seed", 42))
        device = torch.device("cuda" if (cfg.get("GPU_ID", "") != "" and torch.cuda.is_available()) else "cpu")
        train_loader, val_loader, num_classes = build_data(cfg)
        model = build_model(cfg, num_classes).to(device)
        # 权重选择
        weights = getattr(args, "weights", None)
        if not weights:
            for cand in (Path(cfg["save_dir"]) / "best.pt", Path(cfg["save_dir"]) / "last.pt"):
                if cand.exists():
                    weights = str(cand); break
        if not weights:
            raise FileNotFoundError("未找到可用权重，请通过 --weights 指定，或先训练得到 best.pt/last.pt。")
        model.load_state_dict(torch.load(weights, map_location=device))
        criterion = SSDLoss(num_classes=num_classes, alpha=float(cfg.get("ssd_alpha", 1.0)))
        _ = _validate(model, val_loader, criterion, device)
    elif args.cmd == "predict":
        cmd_predict(cfg, images_dir=args.images, out_dir=args.out, weights=getattr(args, "weights", None))
    elif args.cmd == "export-onnx":
        cmd_export_onnx(cfg,
                        weights=getattr(args, "weights", None),
                        out_path=args.out,
                        opset=args.opset,
                        dynamic=args.dynamic,
                        verify=args.verify)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
