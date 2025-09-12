# -*- coding: utf-8 -*-
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


# --- 通用工具函数 ---
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


class BaseCLI:
    """
    一个可扩展的、面向对象的命令行接口基类。
    子类需要实现 build_model, build_data, get_trainer_class 等方法，
    并可以通过重写 add_subparsers_hook 来添加自定义命令或参数。
    """

    def __init__(self):
        self.parser = self._build_parser()

    def run(self):
        """ 主执行函数 """
        args = self.parser.parse_args()

        # 1. 加载和合并配置
        if not Path(args.config).exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        base_cfg = load_json(args.config)
        override_cfg = self._get_override_cfg(args)
        cfg = merge_cfg(base_cfg, override_cfg)

        # 2. 根据命令分发任务
        cmd_map = {
            "train": self.cmd_train,
            "eval": self.cmd_eval,
            "predict": self.cmd_predict,
            "export-onnx": self.cmd_export_onnx,
            "show-config": lambda cfg, args: print(json.dumps(cfg, ensure_ascii=False, indent=2)),
        }

        if args.cmd in cmd_map:
            cmd_map[args.cmd](cfg, args)
        else:
            self.parser.print_help()

    # --- CLI 命令实现 (通用逻辑) ---
    def cmd_train(self, cfg: Dict, args: argparse.Namespace):
        core.init(cfg)
        set_seed(cfg.get("random_seed", 42))
        ensure_dir(cfg["save_dir"])

        data_loader, num_classes = self.build_data(cfg)
        model = self.build_model(cfg, num_classes)

        trainer_class = self.get_trainer_class()
        trainer = trainer_class(cfg, model)

        # 加载预训练权重 (通用逻辑)
        self._try_load_weights(cfg, trainer.model)

        trainer.train(data_loader['train'], data_loader['val'])

    def cmd_eval(self, cfg: Dict, args: argparse.Namespace):
        # ... (通用的 eval 逻辑) ...
        data_loader, num_classes = self.build_data(cfg)
        model = self.build_model(cfg, num_classes)
        trainer = self.get_trainer_class()(cfg, model)
        weights = self._resolve_weights_path(cfg, args.weights)
        trainer.modelLoad(weights)
        trainer.evaluate(data_loader['val'])

    def cmd_predict(self, cfg: Dict, args: argparse.Namespace):

    # ... (通用的 predict 逻辑) ...
    # (这部分可以根据需要进一步抽象)

    def cmd_export_onnx(self, cfg: Dict, args: argparse.Namespace):

    # ... (通用的 onnx 导出逻辑) ...

    # --- 需要子类实现的抽象方法 ---
    def build_model(self, cfg: Dict, num_classes: int) -> torch.nn.Module:
        raise NotImplementedError

    def build_data(self, cfg: Dict) -> Tuple[Dict[str, torch.utils.data.DataLoader], int]:
        raise NotImplementedError

    def get_trainer_class(self) -> type:
        raise NotImplementedError

    # --- 私有和钩子方法 ---
    def _build_parser(self) -> argparse.ArgumentParser:
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
        p.add_argument("--config", type=str, default="configs/movenet_config.json", help="路径: JSON 配置文件")
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

    def add_subparsers_hook(self, subparsers: argparse._SubParsersAction):
        """ (可选) 子类可以重写此方法来添加或修改子命令参数 """
        pass

    def _get_override_cfg(self, args: argparse.Namespace) -> Dict:
        """ 从命令行参数中收集需要覆盖配置的项 """
        # (这里粘贴 merge_cfg 的逻辑)
        return {...}

    def _resolve_weights_path(self, cfg: Dict, weights_arg: str) -> str:
# (这里粘贴自动寻找 best.pt / last.pt 的逻辑)