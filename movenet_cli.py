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

import os
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

import core
from core.dataloader.simple_loader import SimpleImageFolder
from core.task.task import Task
from core.models.movenet import MoveNet
from core.models.dummy_movenet import DummyMoveNet
from core.dataloader.dataloader import CoCo2017DataLoader


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

    # 构造MoveNet模型
    model = MoveNet(
        num_joints=cfg["num_classes"],
        width_mult=cfg.get("width_mult", 1.0),
        neck_outc=cfg.get("neck_outc", 64),
        head_midc=cfg.get("head_midc", 32),
        backbone=cfg.get("backbone", "mobilenet_v2"),
    )
    return model

def build_data(cfg: Dict[str, Any]):
    return CoCo2017DataLoader(cfg)

# ----------------------------------------------------------------------
# 子命令：train
# ----------------------------------------------------------------------
def cmd_train(cfg: Dict[str, Any]):
    core.init(cfg)
    set_seed(cfg.get("random_seed", 42))
    ensure_dir(cfg["save_dir"])

    model: torch.nn.Module = build_model(cfg)
    data: CoCo2017DataLoader = build_data(cfg)
    train_loader, val_loader = data.getTrainValDataloader()

    # 创建任务
    task = Task(cfg, model)

    # 如果有先前训练的结果，尝试加载
    try:
        weights = os.path.join(cfg.get("save_dir", "output"), "best.pt")
        if os.path.exists(weights):
            task.modelLoad(weights)
            print(f"[INFO] Find the pre-trained weights file, load it and continue the training process")
    except:
        print(f"[INFO] Training a model from scratch")

    # 开始训练任务
    task.train(train_loader, val_loader)

# ----------------------------------------------------------------------
# 子命令：eval
# ----------------------------------------------------------------------
def cmd_eval(cfg: Dict[str, Any], weights: str):
    core.init(cfg)
    set_seed(cfg.get("random_seed", 42))

    model = build_model(cfg)
    data = build_data(cfg)
    _, val_loader = data.getTrainValDataloader()

    task = Task(cfg, model)
    if not weights:
        weights = str(Path(cfg["save_dir"]) / "best.pt")
    task.modelLoad(weights)
    task.evaluate(val_loader)

# ----------------------------------------------------------------------
# 子命令：predict（简单演示，复用你 Task.predict 时的可视化）
# ----------------------------------------------------------------------
def cmd_predict(cfg, images_dir, out_dir, weights=None):
    core.init(cfg)  # 和 train/eval 一致的初始化
    os.makedirs(out_dir, exist_ok=True)

    # 1) 构建模型 & Task
    model = build_model(cfg)
    task = Task(cfg, model)

    # 2) 选择权重：--weights > best.pt > last.pt
    if not weights:
        cand_best = Path(cfg["save_dir"]) / "best.pt"
        cand_last = Path(cfg["save_dir"]) / "last.pt"
        if cand_best.exists():
            weights = str(cand_best)
        elif cand_last.exists():
            weights = str(cand_last)
        else:
            raise FileNotFoundError("未找到可用权重，请用 --weights 指定，或先训练得到 best.pt/last.pt。")
    task.modelLoad(weights)  # ←←← 关键：加载训练好的权重

    # 3) 用 SimpleImageFolder 做 DataLoader，然后直接用 Task.predict 输出
    dataset = SimpleImageFolder(images_dir, img_size=cfg["img_size"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    task.predict(loader, out_dir)   # 会输出和旧版一致的 1+4 张图

    
# ----------------------------------------------------------------------
# 子命令：export-onnx
# ----------------------------------------------------------------------
def cmd_export_onnx(cfg: Dict[str, Any], weights: str, out_path: str,
                    opset: int, dynamic: bool, verify: bool):
    """
    导出 ONNX：
      - 默认导出 4 个头（heatmaps/centers/regs/offsets）
      - 若 cfg["export_keypoints"] 为 True，则导出关键点版单头 [B, 51]，即 (x,y,score)*17
        - 坐标为归一化到 0~1
        - score 为对应关节 heatmap 处的峰值
    """
    # 这些工具函数/类按你工程里的位置来
    core.init(cfg)
    set_seed(cfg.get("random_seed", 42))
    ensure_dir(Path(out_path).parent)

    model = build_model(cfg)
    task = Task(cfg, model)

    # 自动挑选权重：best.pt 优先，否则 last.pt
    if not weights:
        for c in (Path(cfg["save_dir"]) / "best.pt", Path(cfg["save_dir"]) / "last.pt"):
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

    # # 需要 onnx
    # try:
    #     import onnx  # noqa: F401
    # except Exception:
    #     raise RuntimeError("缺少 onnx，请先安装：pip install onnx")

    use_keypoints = bool(cfg.get("export_keypoints", False))
    print(f"[INFO] Exporting to ONNX: {out_path} (opset={opset}, dynamic={dynamic}, keypoints={use_keypoints})")

    if use_keypoints:
        # --- 关键点版导出：单一输出 [B, 51] ---
        # 惰性导入，以免训练路径无该依赖时报错
        wrapper = DummyMoveNet(
            movenet=task.model,
            num_joints=int(cfg.get("num_classes", 17)),
            img_size=int(cfg.get("img_size", 192)),
            target_stride=int(cfg.get("target_stride", 4)),
            hm_th=float(cfg.get("hm_th", 0.1)),
        ).to(device).eval()

        out_names = ["keypoints"]
        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                "input":     {0: "batch", 2: "height", 3: "width"},
                "keypoints": {0: "batch"},  # [B, 51]
            }

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

    else:
        # --- 保持原有 4 头导出（便于外部后处理） ---
        class _Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__();
                self.m = m
            def forward(self, x):
                y = self.m(x)  # dict
                return y["heatmaps"], y["centers"], y["regs"], y["offsets"]

        wrapper = _Wrapper(task.model).to(device).eval()

        out_names = ["heatmaps", "centers", "regs", "offsets"]
        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                "input":   {0: "batch", 2: "height", 3: "width"},
                "heatmaps":{0: "batch", 2: "h", 3: "w"},
                "centers": {0: "batch", 2: "h", 3: "w"},
                "regs":    {0: "batch", 2: "h", 3: "w"},
                "offsets": {0: "batch", 2: "h", 3: "w"},
            }

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
            if use_keypoints:
                assert len(outs) == 1 and outs[0].ndim == 2 and outs[0].shape[1] == int(cfg.get("num_classes", 17)) * 3
            else:
                assert len(outs) == 4 and all(isinstance(o, np.ndarray) for o in outs)
            print("[OK] ONNXRuntime quick check passed.")
        except Exception as e:
            print(f"[WARN] onnxruntime 验证失败（可忽略）：{e}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser():
    epilog_msg = """
示例:
  # 使用默认配置训练
  python movenet_cli.py --config config.json train --epochs 50 --batch-size 64

  # 在验证集上评估
  python movenet_cli.py --config config.json eval --weights output/best.pt

  # 对目录中的图片预测并可视化到 output_vis/
  python movenet_cli.py --config config.json predict --images ./test_imgs --out ./output_vis

  # 导出 ONNX (默认4头)
  python movenet_cli.py --config config.json export-onnx --out output/movenet.onnx

  # 导出关键点版 ONNX (单一输出 [B,51])
  python movenet_cli.py --config config.json export-onnx --keypoints --out output/movenet_kps.onnx

  # 打印合并后的配置
  python movenet_cli.py --config config.json show-config
"""
    p = argparse.ArgumentParser(
        description="MoveNet Unified CLI (train/eval/predict/export/show-config)",
        epilog=epilog_msg,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=str, default="config.json", help="路径: JSON 配置文件")
    p.add_argument("--save-dir", type=str, help="覆盖 cfg.save_dir (保存权重与结果)")
    p.add_argument("--img-size", type=int, help="覆盖 cfg.img_size (输入图像大小, 默认192)")
    p.add_argument("--num-classes", type=int, help="覆盖 cfg.num_classes (关键点个数, 默认17)")
    p.add_argument("--width-mult", type=float, help="覆盖 cfg.width_mult (backbone 宽度倍率, 默认1.0)")
    p.add_argument("--backbone", type=str, help="覆盖 cfg.backbone (mobilenet_v2 / shufflenet_v2)")
    p.add_argument("--gpu-id", type=str, help="覆盖 cfg.GPU_ID (''=CPU, '0'=第0张GPU)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # ========== train ==========
    sp_tr = sub.add_parser("train", help="训练模型并保存权重 (last.pt / best.pt)")
    sp_tr.add_argument("--epochs", type=int, help="训练轮数 (覆盖配置文件)")
    sp_tr.add_argument("--batch-size", type=int, help="批大小 (覆盖配置文件)")
    sp_tr.add_argument("--lr", type=float, help="学习率 (覆盖配置文件)")
    sp_tr.add_argument("--optimizer", type=str, help="优化器 (Adam / SGD)")
    sp_tr.add_argument("--scheduler", type=str, help="学习率调度器, 如 'MultiStepLR-70,100-0.1'")

    # ========== eval ==========
    sp_ev = sub.add_parser("eval", help="在验证集上评估模型")
    sp_ev.add_argument("--weights", type=str, help="权重路径 (默认 save_dir/best.pt)")

   # ========== predict ==========
    sp_pr = sub.add_parser("predict", help="对目录中的图片进行预测与可视化")
    sp_pr.add_argument("--images", type=str, required=True, help="输入图片目录")
    sp_pr.add_argument("--out", type=str, required=True, help="输出目录 (保存可视化结果)")
    sp_pr.add_argument("--weights", type=str, help="权重路径 (默认 save_dir/best.pt 或 last.pt)")

    # ========== export-onnx ==========
    sp_ex = sub.add_parser("export-onnx", help="导出 ONNX 模型文件")
    sp_ex.add_argument("--weights", type=str, help="权重路径 (默认 best.pt / last.pt)")
    sp_ex.add_argument("--out", type=str, default="output/movenet.onnx", help="导出 ONNX 文件路径")
    sp_ex.add_argument("--opset", type=int, default=13, help="ONNX opset 版本 (默认 13)")
    sp_ex.add_argument("--dynamic", action="store_true", help="导出动态 batch/height/width")
    sp_ex.add_argument("--verify", action="store_true", help="导出后用 onnxruntime 进行推理校验")
    sp_ex.add_argument("--keypoints", action="store_true",
                       help="导出关键点版 ONNX (输出[B,51]=(x,y,score)*17)")

    # ========== show-config ==========
    sub.add_parser("show-config", help="打印最终合并后的配置 (JSON 格式)")

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
        cmd_predict(cfg, images_dir=args.images, out_dir=args.out,
                    weights=getattr(args, "weights", None))
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
