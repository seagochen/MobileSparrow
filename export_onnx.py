#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn

# 你的工程内导入
from config import cfg
from lib import init, MoveNet, Task  # 与 train 时保持一致

class ExportWrapper(nn.Module):
    """
    将 dict 输出包装为 tuple： (heatmaps, centers, regs, offsets)
    这样 torch.onnx.export 更稳。
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return (out["heatmaps"], out["centers"], out["regs"], out["offsets"])
        # 兼容 list/tuple
        return tuple(out)

def main():
    init(cfg)

    # —— 与训练时一致的建模参数 —— #
    model = MoveNet(
        num_joints=cfg["num_classes"],
        width_mult=cfg["width_mult"],        # 你保留了 width_mult，这里照用
        # backbone="mobilenet_v2",           # 如你在 MoveNet 里开放了这个参数，可显式指定
        # 其他必要的构造参数……
    )

    # 加载权重（best.pt 或 last.pt）
    weights_path = os.path.join(cfg["save_dir"], "best.pt")  # 或 "last.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = Task(cfg, model)                # 复用你现有的加载逻辑
    task.modelLoad(weights_path, data_parallel=False)
    task.model.eval().to(device)

    # 包装成顺序输出
    export_model = ExportWrapper(task.model).eval().to(device)

    # dummy 输入（与训练分辨率一致）
    img_h = cfg.get("img_size", 192)
    img_w = cfg.get("img_size", 192)
    dummy = torch.randn(1, 3, img_h, img_w, device=device)

    # 导出信息
    input_names  = ["input"]
    output_names = ["heatmaps", "centers", "regs", "offsets"]

    # 动态 batch（空间分辨率也可设成动态，但很多部署链路更喜欢固定 HxW，这里只放开 batch）
    dynamic_axes = {
        "input":    {0: "batch"},
        "heatmaps": {0: "batch"},
        "centers":  {0: "batch"},
        "regs":     {0: "batch"},
        "offsets":  {0: "batch"},
    }

    onnx_path = os.path.join(cfg["save_dir"], "movenet_pytorch.onnx")
    torch.onnx.export(
        export_model,
        dummy,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=12,               # 11 也行，通常建议 12/13
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        verbose=False
    )

    print(f"[OK] ONNX 导出完成: {onnx_path}")

if __name__ == "__main__":
    main()
