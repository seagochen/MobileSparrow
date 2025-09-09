#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified CLI for MoveNet project:
  - train:     训练，保存 last.pt / best.pt
  - eval:      在验证集上评估
  - predict:   目录/列表预测并可视化
  - export-onnx: 导出 ONNX（4 个命名输出）
  - show-config: 打印合并后的最终配置

依赖：PyTorch、opencv-python、numpy、onnx(导出时)、onnxruntime(可选验证)
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from core import init, CoCo2017DataLoader, MoveNet, Task


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def merge_cfg(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(base)
    for k, v in override.items():
        if v is not None:
            cfg[k] = v
    return cfg

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 组装模型与数据
# ----------------------------------------------------------------------
def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    # 你新的 MoveNet 构造签名
    # 例：MoveNet(backbone=..., num_joints=..., width_mult=...)
    model = MoveNet(
        num_joints=cfg["num_classes"],
        width_mult=cfg.get("width_mult", 1.0),
        backbone=cfg.get("backbone", "mobilenet_v2"),
    )
    return model

def build_data(cfg: Dict[str, Any]):
    return CoCo2017DataLoader(cfg)

# ----------------------------------------------------------------------
# 子命令：train
# ----------------------------------------------------------------------
def cmd_train(cfg: Dict[str, Any]):
    init(cfg)
    set_seed(cfg.get("random_seed", 42))
    ensure_dir(cfg["save_dir"])

    model = build_model(cfg)
    data = build_data(cfg)
    train_loader, val_loader = data.getTrainValDataloader()

    task = Task(cfg, model)
    task.train(train_loader, val_loader)   # 你之前已改过只保留 best.pt/last.pt

# ----------------------------------------------------------------------
# 子命令：eval
# ----------------------------------------------------------------------
def cmd_eval(cfg: Dict[str, Any], weights: str):
    init(cfg)
    set_seed(cfg.get("random_seed", 42))

    model = build_model(cfg)
    data = build_data(cfg)
    val_loader = data.getEvalDataloader()

    task = Task(cfg, model)
    if not weights:
        weights = str(Path(cfg["save_dir"]) / "best.pt")
    task.modelLoad(weights)
    task.evaluate(val_loader)

# ----------------------------------------------------------------------
# 子命令：predict（简单演示，复用你 Task.predict 时的可视化）
# ----------------------------------------------------------------------
def cmd_predict(cfg: Dict[str, Any], images_dir: str, out_dir: str):
    init(cfg)
    set_seed(cfg.get("random_seed", 42))
    ensure_dir(out_dir)

    model = build_model(cfg)
    task = Task(cfg, model)
    # 默认加载 best.pt
    weights = str(Path(cfg["save_dir"]) / "best.pt")
    task.modelLoad(weights)

    data = build_data(cfg)
    loader = data.getTestDataloader(images_dir)  # 如无此函数，可自行按你项目补齐
    task.predict(loader, out_dir)

# ----------------------------------------------------------------------
# 子命令：export-onnx
# ----------------------------------------------------------------------
def cmd_export_onnx(cfg: Dict[str, Any], weights: str, out_path: str,
                    opset: int, dynamic: bool, verify: bool):
    init(cfg)
    set_seed(cfg.get("random_seed", 42))
    ensure_dir(Path(out_path).parent)

    model = build_model(cfg)
    task = Task(cfg, model)

    if not weights:
        # 优先 best.pt，否则 last.pt
        candidates = [Path(cfg["save_dir"]) / "best.pt", Path(cfg["save_dir"]) / "last.pt"]
        for c in candidates:
            if c.exists():
                weights = str(c)
                break
    if not weights:
        raise FileNotFoundError("未找到可用权重，请通过 --weights 指定，或先训练得到 best.pt/last.pt。")

    task.modelLoad(weights)
    task.model.eval()
    device = torch.device("cuda" if (cfg.get("GPU_ID", "") != "" and torch.cuda.is_available()) else "cpu")
    task.model.to(device)

    h = w = int(cfg.get("img_size", 192))
    dummy = torch.randn(1, 3, h, w, device=device)

    # 约定：MoveNet.forward 返回 dict
    # 导出 4 个命名输出：heatmaps / centers / regs / offsets
    print(f"[INFO] Exporting to ONNX: {out_path} (opset={opset}, dynamic={dynamic})")
    out_names = ["heatmaps", "centers", "regs", "offsets"]
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "heatmaps": {0: "batch", 2: "h", 3: "w"},
            "centers":  {0: "batch", 2: "h", 3: "w"},
            "regs":     {0: "batch", 2: "h", 3: "w"},
            "offsets":  {0: "batch", 2: "h", 3: "w"},
        }

    # 需要 onnx
    try:
        import onnx  # noqa: F401
    except Exception:
        raise RuntimeError("缺少 onnx，请先安装：pip install onnx")

    # 包一层 wrapper 把 dict 按固定顺序展开
    class _Wrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            y = self.m(x)
            return y["heatmaps"], y["centers"], y["regs"], y["offsets"]

    wrapper = _Wrapper(task.model).to(device).eval()

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["input"],
        output_names=out_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    print(f"[OK] ONNX saved to: {out_path}")

    if verify:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(out_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
            outs = sess.run(None, {"input": dummy.detach().cpu().numpy()})
            assert len(outs) == 4 and all(isinstance(o, np.ndarray) for o in outs)
            print("[OK] ONNXRuntime quick check passed.")
        except Exception as e:
            print(f"[WARN] onnxruntime 验证失败（可忽略）：{e}")

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="MoveNet Unified CLI")
    p.add_argument("--config", type=str, default="config.json", help="路径：JSON 配置文件")
    p.add_argument("--save-dir", type=str, help="覆盖 cfg.save_dir")
    p.add_argument("--img-size", type=int, help="覆盖 cfg.img_size")
    p.add_argument("--num-classes", type=int, help="覆盖 cfg.num_classes")
    p.add_argument("--width-mult", type=float, help="覆盖 cfg.width_mult")
    p.add_argument("--backbone", type=str, help="覆盖 cfg.backbone (mobilenet_v2 / shufflenet_v2)")
    p.add_argument("--gpu-id", type=str, help="覆盖 cfg.GPU_ID（''=CPU）")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp_tr = sub.add_parser("train", help="训练")
    # 训练超参也可用 CLI 覆盖
    sp_tr.add_argument("--epochs", type=int)
    sp_tr.add_argument("--batch-size", type=int)
    sp_tr.add_argument("--lr", type=float)
    sp_tr.add_argument("--optimizer", type=str)
    sp_tr.add_argument("--scheduler", type=str)

    sp_ev = sub.add_parser("eval", help="验证集评估")
    sp_ev.add_argument("--weights", type=str, help="权重路径（默认用 save_dir/best.pt）")

    sp_pr = sub.add_parser("predict", help="目录预测与可视化（需要你 Data 中相应方法）")
    sp_pr.add_argument("--images", type=str, required=True, help="图片目录")
    sp_pr.add_argument("--out", type=str, required=True, help="输出目录（可视化）")

    sp_ex = sub.add_parser("export-onnx", help="导出 ONNX")
    sp_ex.add_argument("--weights", type=str, help="权重（默认 best.pt / last.pt）")
    sp_ex.add_argument("--out", type=str, default="output/core.onnx", help="ONNX 路径")
    sp_ex.add_argument("--opset", type=int, default=13)
    sp_ex.add_argument("--dynamic", action="store_true", help="导出动态高宽/批次")
    sp_ex.add_argument("--verify", action="store_true", help="导出后用 onnxruntime 快速校验")

    sub.add_parser("show-config", help="打印最终配置")
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    # 载入 JSON 配置
    if not Path(args.config).exists():
        raise FileNotFoundError(f"未找到配置文件：{args.config}")
    base_cfg = load_json(args.config)

    # 命令行覆盖项
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
        cmd_eval(cfg, weights=getattr(args, "weights", None))
    elif args.cmd == "predict":
        cmd_predict(cfg, images_dir=args.images, out_dir=args.out)
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
