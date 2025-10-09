#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from typing import Dict, Any, Optional

import torch

from sparrow.trainer.movenet_fpn_sp_trainer import MoveNetSingleTrainer
from sparrow.trainer.sixrepnet_trainer import SixDRepNetTrainer
from sparrow.trainer.ssdlite_fpn_trainer import SSDLiteTrainer


# ----------------------------
# 小工具：读取 YAML（用于给 Trainer 传路径即可）
# ----------------------------
def _ensure_file(path: str, what: str):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")
    return path

def load_model_pt(model: torch.nn.Module, path: Optional[str], strict: bool = False) -> Dict[str, Any] | None:
    """兼容多种 state_dict 断点格式的通用加载。"""
    if not path:
        return None
    ckpt = torch.load(path, map_location="cpu")
    sd = None
    if isinstance(ckpt, dict):
        for k in ("model_state", "ema_state", "model", "state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]; break
        if sd is None and all(isinstance(k, str) and "." in k for k in ckpt.keys()):
            sd = ckpt
    if sd is None:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    model.load_state_dict(sd, strict=strict)
    return ckpt

# ----------------------------
# 任务映射
# ----------------------------
TASKS = {
    "ssdlite": SSDLiteTrainer,
    "movenet": MoveNetSingleTrainer,
    "sixrepnet": SixDRepNetTrainer,
}

# ----------------------------
# 子命令：train
# ----------------------------
def cmd_train(args: argparse.Namespace) -> None:
    _ensure_file(args.config, "YAML config")
    Trainer = TASKS[args.task]
    trainer = Trainer(yaml_path=args.config)

    # 可选：若提供 --weights，先加载模型权重再训练（覆盖初始化权重）
    if args.weights and os.path.isfile(args.weights):
        load_model_pt(trainer.model, args.weights, strict=False)

    trainer.train_model()  # 新 Trainer 的统一训练入口

# ----------------------------
# 子命令：eval
# ----------------------------
def cmd_eval(args: argparse.Namespace) -> None:
    _ensure_file(args.config, "YAML config")
    Trainer = TASKS[args.task]
    trainer = Trainer(yaml_path=args.config)

    # 加载权重（优先 CLI；若未提供，则尝试 save_dir/{best.pt,last.pt}）
    ckpt_path = args.weights
    if not ckpt_path:
        # 尝试从 trainer.save_dir 推断
        last = os.path.join(trainer.save_dir or ".", "last.pt")
        best = os.path.join(trainer.save_dir or ".", "best.pt")
        ckpt_path = best if os.path.isfile(best) else (last if os.path.isfile(last) else "")

    if ckpt_path:
        load_model_pt(trainer.model, ckpt_path, strict=False)

    # 统一调用 BaseTrainer 的 evaluate 签名
    # 注意：有的任务（如 SixDRepNet）会返回 (loss_dict, metrics_dict)
    out = trainer.evaluate(
        model=trainer.model,
        loss_fn=trainer.loss_fn,
        epoch=0,
        loader=trainer.val_dl,
        device=trainer.device,
    )
    if isinstance(out, tuple):
        loss_dict, metric_dict = out
        print("[eval] loss:", loss_dict)
        print("[eval] metrics:", metric_dict)
    else:
        print("[eval] metrics:", out)

# ----------------------------
# 子命令：export
# ----------------------------
def cmd_export(args: argparse.Namespace) -> None:
    _ensure_file(args.config, "YAML config")
    Trainer = TASKS[args.task]
    trainer = Trainer(yaml_path=args.config)

    # 加载权重
    if args.weights:
        load_model_pt(trainer.model, args.weights, strict=False)
    trainer.model.eval()

    # 优先调用任务自带导出（若已实现）
    if hasattr(trainer, "export_onnx"):
        try:
            trainer.export_onnx()  # 某些任务可能内部读取形状/保存路径
            return
        except NotImplementedError:
            pass
        except Exception as e:
            print(f"[export] task-level exporter failed, fallback to generic ONNX: {e}")

    # 兜底：最小 ONNX 导出
    b, c, h, w = (int(x) for x in args.input_shape.split(",")) if args.input_shape else (1, 3, 320, 320)
    dummy = torch.randn(b, c, h, w)
    if args.half:
        trainer.model.half(); dummy = dummy.half()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    dynamic_axes = {"input": {0: "batch"}} if args.dynamic_batch else None
    torch.onnx.export(
        trainer.model, (dummy,), args.output,
        input_names=["input"], output_names=["output"],
        opset_version=args.opset, do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    print(f"=> ONNX saved to: {args.output}")

# ----------------------------
# CLI
# ----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Sparrow CLI (new-trainer)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def base(sp: argparse.ArgumentParser):
        sp.add_argument("-c", "--config", required=True, help="YAML 配置文件路径")
        sp.add_argument("-t", "--task", required=True, choices=TASKS.keys(), help="任务类型")
        sp.add_argument("--weights", default="", help="权重文件（可选）")

    sp_tr = sub.add_parser("train", help="训练")
    base(sp_tr)

    sp_ev = sub.add_parser("eval", help="评估")
    base(sp_ev)

    sp_ex = sub.add_parser("export", help="导出 ONNX")
    base(sp_ex)
    sp_ex.add_argument("--output", default="model.onnx", help="ONNX 输出路径")
    sp_ex.add_argument("--input-shape", default="1,3,320,320", help="dummy 输入形状 (b,c,h,w)")
    sp_ex.add_argument("--opset", type=int, default=13, help="ONNX opset")
    sp_ex.add_argument("--half", action="store_true", help="半精度导出")
    sp_ex.add_argument("--dynamic-batch", action="store_true", help="开启动态 batch 维")

    return p

def main() -> None:
    args = build_argparser().parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "export":
        cmd_export(args)

if __name__ == "__main__":
    main()
