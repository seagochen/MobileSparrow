# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# 通用工具函数
# =========================
PathLike = Union[str, Path]


def load_json(path: PathLike) -> Dict[str, Any]:
    """读取 JSON 文件为 dict。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: PathLike) -> None:
    """将 dict 保存为 JSON 文件（含缩进/不转义中文）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def merge_cfg(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """浅合并配置：仅当 override[k] 非 None 时覆盖。"""
    cfg = dict(base)
    for k, v in override.items():
        if v is not None:
            cfg[k] = v
    return cfg


def ensure_dir(p: PathLike) -> None:
    """确保目录存在。"""
    Path(p).mkdir(parents=True, exist_ok=True)


# =========================
# 随机性控制 / 复现
# =========================
def seed_everything(seed: int) -> None:
    """
    轻量版：常规训练推荐。保留 cuDNN 自适应以求速度。
    说明：评测/对比时再切换到 strict 版本。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 常规训练：允许 cuDNN 算法搜索（通常更快）
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def seed_everything_strict(seed: int) -> None:
    """
    严格版：用于调试/论文复现/审计。更稳定但可能更慢。
    提示：PYTHONHASHSEED 需在 Python 启动前设置才真正生效：
      Linux/macOS: PYTHONHASHSEED=0 python train.py --seed 42
      Windows(cmd): set PYTHONHASHSEED=0 && python train.py --seed 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN 确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁止自动算法搜索，避免波动

    # 使用非确定性算子时直接报错，便于定位
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    # （可选）关闭 TF32，减少数值误差来源（按需启用）
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


# =========================
# 模型保存 / 加载（.pt）
# =========================
def save_model_pt(
    model: nn.Module,
    path: PathLike,
    optimizer: Optional[optim.Optimizer] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    save_state_dict: bool = True,
) -> None:
    """
    保存模型到 .pt 文件（推荐 save_state_dict=True 保存检查点字典）。
    - model: 要保存的模型
    - path: 目标文件路径（.pt）
    - optimizer: 可选，保存优化器状态
    - epoch: 可选，当前 epoch 号
    - extra: 可选，自定义元信息（超参、指标等）
    - save_state_dict: True=仅保存 state_dict（推荐）；False=保存整个模型（不推荐，依赖源码/类名）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if save_state_dict:
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "extra": extra or {},
            "torch_version": torch.__version__,
        }
        torch.save(ckpt, path)
    else:
        # 保存整个模型（包含结构）。不推荐：依赖 pickle 和源码位置，跨环境易出问题。
        torch.save(
            {
                "model_full": model,  # 警告：反序列化需信任来源！
                "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                "epoch": epoch,
                "extra": extra or {},
                "torch_version": torch.__version__,
            },
            path,
        )


def load_model_pt(
    path: PathLike,
    model: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    """
    从 .pt 加载模型/检查点。
    - path: .pt 文件
    - model: 已实例化的 nn.Module；若 None 且文件保存了 'model_full'，将直接返回该完整模型
    - optimizer: 可选，若提供且检查点包含优化器状态则会恢复
    - map_location: 'cpu' / 'cuda' / torch.device(...)
    - strict: load_state_dict 的 strict 选项

    返回:
      (model或None, meta字典)；meta 包含 epoch/extra/torch_version 等。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {str(path)}")

    # 注意：torch.load 使用 pickle，需确保文件可信！
    ckpt = torch.load(path, map_location=map_location)

    meta = {"epoch": None, "extra": {}, "torch_version": None}

    # 1) 推荐格式：检查点字典
    if isinstance(ckpt, dict):
        # 优先处理 state_dict
        state = ckpt.get("model_state") or ckpt.get("state_dict")
        meta["epoch"] = ckpt.get("epoch")
        meta["extra"] = ckpt.get("extra", {})
        meta["torch_version"] = ckpt.get("torch_version")

        # 如果包含完整模型且用户未传入实例，则直接返回完整模型
        if model is None and "model_full" in ckpt:
            model = ckpt["model_full"]

        # 否则尝试把 state_dict 加载到传入的 model
        if state is not None and model is not None:
            missing, unexpected = model.load_state_dict(state, strict=strict)
            # PyTorch < 2.0 返回的是命名列表；>=2.0 可能返回错误/None，这里统一兼容
            if isinstance(missing, (list, tuple)) and (missing or unexpected):
                print(
                    f"[load_model_pt] Warning - missing keys: {missing}, "
                    f"unexpected keys: {unexpected}"
                )

        # 恢复优化器
        if optimizer is not None and ckpt.get("optimizer_state") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                print(f"[load_model_pt] Warning - failed to load optimizer state: {e}")

        return model, meta

    # 2) 兼容：直接保存的 state_dict
    if isinstance(ckpt, (dict,)):
        if model is None:
            raise ValueError("A model instance must be provided to load a bare state_dict.")
        model.load_state_dict(ckpt, strict=strict)
        return model, meta

    # 3) 兼容：直接保存的完整模型对象
    if isinstance(ckpt, nn.Module):
        if model is None:
            model = ckpt
            return model, meta
        else:
            # 用户传了 model，又加载到一个完整模型 -> 按 state_dict 路径尝试
            model.load_state_dict(ckpt.state_dict(), strict=strict)
            return model, meta

    # 其他未知格式
    raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")
