#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Sparrow CLI（中文说明）

本文件提供基于 YAML 配置的命令行入口，统一完成以下任务：
1) 从 YAML 构建模型 / 训练器 / 数据加载器（支持别名 ALIASES、完整路径、文件 URI）
2) 训练：断点继续、加载权重、日志记录
3) 评估：按相同配置执行验证/评估过程
4) 导出：导出为 ONNX / TorchScript（可选 wrapper 适配层）

基本用法：
  训练：python sparrow_cli.py train -c configs/ssdlite.yaml
  评估：python sparrow_cli.py eval  -c configs/ssdlite.yaml --weights outputs/xxx/best.pt
  导出：python sparrow_cli.py export -c configs/ssdlite.yaml --weights outputs/xxx/best.pt --format onnx --output out.onnx

配置说明（简要）：
  model:   指定模型类与其参数
  trainer: 指定训练器类与其参数（优化器/调度/AMP/EMA/保存策略等）
  data:    指定 train/val 的构建函数和参数（dataset_root、batch_size 等）

注意：
- ALIASES（别名映射）可在本文件顶部修改，便于短名引用。
- 也可使用完整点号路径 "pkg.mod:Class" 或文件 URI "file:/abs/path.py#Class"。
- CLI 支持 --set k=v 的临时覆盖（点号路径），如：--set trainer.args.epochs=5
"""

import argparse
import importlib
import inspect
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import yaml

# =========================================================
# 别名映射（短名 -> 真正的可导入路径）
# - 你可以在此处增删改别名，方便 YAML 用短名指向你的类/函数
# =========================================================
ALIASES: Dict[str, str] = {
    # 模型
    "movenet": "sparrow.models.movenet.MoveNet",
    "ssdlite": "sparrow.models.ssdlite.SSDLite",

    # 训练器
    "kpts_trainer": "sparrow.task.kpts_trainer.KptsTrainer",
    "dets_trainer": "sparrow.task.dets_trainer.DetsTrainer",

    # 数据构建器（dataloader 工厂函数）
    "coco_kpts_dataloader": "sparrow.datasets.coco_kpts.create_kpts_dataloader",
    "coco_dets_dataloader": "sparrow.datasets.coco_dets.create_dets_dataloader",
    "simple_image_folder": "sparrow.datasets.simple_loader.SimpleImageFolder",

    # 导出包装器
    "dummy_movenet": "sparrow.models.onnx.dummy_movenet.DummyMoveNet",
    "dummy_ssdlite": "sparrow.models.onnx.dummy_ssdlite.DummySSDLite",
}

# =========================================================
# 基础工具：解析字符串为可调用对象 / 安全实例化
# =========================================================
def resolve_callable(spec: str) -> Callable[..., Any]:
    """
    将字符串规范化为可调用对象（类或函数），支持三种写法：
      1) 别名：在 ALIASES 中注册的短名，例如 "ssdlite"
      2) 完整路径："pkg.mod:Attr" 或 "pkg.mod.Attr"
      3) 文件 URI："file:/abs/path/to/module.py#Attr"

    参数：
      spec (str): 目标的字符串表示

    返回：
      callable: 已解析的类或函数对象

    异常：
      - ValueError：当字符串格式不合法或对象不存在
      - ImportError / ModuleNotFoundError：当模块导入失败
    """
    if not isinstance(spec, str) or not spec:
        raise ValueError(f"不可解析的可调用对象标识: {spec!r}")

    # 1) 别名
    if spec in ALIASES:
        spec = ALIASES[spec]

    # 2) 文件 URI：file:/abs/path.py#Attr
    if spec.startswith("file:"):
        # 支持 file:/xxx.py#ClassName 或 file:/xxx.py:ClassName
        path, _, attr = spec[5:].replace(":", "#", 1).partition("#")
        if not attr:
            raise ValueError(f"文件 URI 缺少对象名: {spec}")
        module = _import_from_file(path)
        if not hasattr(module, attr):
            raise ImportError(f"在模块 {path} 中未找到对象 {attr}")
        obj = getattr(module, attr)
        if not callable(obj):
            raise ValueError(f"对象 {attr} 不是可调用的: {obj!r}")
        return obj

    # 3) 完整路径：pkg.mod:Attr 或 pkg.mod.Attr
    if ":" in spec:
        mod_name, attr = spec.split(":", 1)
    else:
        # 尝试最后一个点作为分隔
        parts = spec.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"无法解析的路径: {spec!r}")
        mod_name, attr = parts

    module = importlib.import_module(mod_name)
    if not hasattr(module, attr):
        raise ImportError(f"在模块 {mod_name} 中未找到对象 {attr}")
    obj = getattr(module, attr)
    if not callable(obj):
        raise ValueError(f"对象 {attr} 不是可调用的: {obj!r}")
    return obj


def _import_from_file(path: str) -> types.ModuleType:
    """
    从绝对路径导入一个 .py 文件为模块（不需要在 sys.path 中）。
    仅用于开发/试验场景；生产建议使用“包路径”方式。
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    mod_name = f"_dyn_{Path(path).stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从文件导入模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


def instantiate_with_filtered_kwargs(cls_or_fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Any:
    """
    安全实例化辅助：根据构造函数签名过滤掉 kwargs 中无用的键，避免“unexpected keyword argument”错误。

    参数：
      cls_or_fn: 目标类或函数（可调用）
      kwargs (dict): 原始参数字典

    返回：
      实例对象或函数返回值（取决于 cls_or_fn ）
    """
    sig = inspect.signature(cls_or_fn)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if (k in accepted or any(
        p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        for p in sig.parameters.values()
    ))}
    return cls_or_fn(**filtered)


def maybe_instantiate(callable_or_obj: Any, kwargs: Optional[Dict[str, Any]] = None) -> Any:
    """
    根据入参类型决定是否实例化：
      - 若传入的是“类/函数”，则用 kwargs 实例化并返回实例
      - 若传入的是“已构建对象”，则直接返回对象本身

    典型用途：
      - 用于统一处理 model/trainer/builder/wrapper 等对象的构建

    注意：
      - 若构造函数不接受 kwargs 中的某些键，建议使用
        `instantiate_with_filtered_kwargs()` 先按签名过滤
    """
    if kwargs is None:
        kwargs = {}
    if inspect.isclass(callable_or_obj) or inspect.isfunction(callable_or_obj):
        return instantiate_with_filtered_kwargs(callable_or_obj, kwargs)
    return callable_or_obj


# =========================================================
# YAML 读取与 CLI 覆盖
# =========================================================
def load_yaml(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件为字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_by_path(dct: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """按点号路径写入字典：a.b.c = value"""
    keys = dotted_key.split(".")
    cur = dct
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _parse_cli_overrides(pairs: Optional[list[str]]) -> Dict[str, Any]:
    """
    解析 --set a.b.c=val 风格的 CLI 覆盖项，支持 int/float/bool/None 基础类型自动识别。
    """
    if not pairs:
        return {}
    out: Dict[str, Any] = {}
    for kv in pairs:
        if "=" not in kv:
            raise ValueError(f"--set 需要 k=v 形式，收到: {kv}")
        k, v = kv.split("=", 1)
        v = _auto_cast(v)
        out[k] = v
    return out


def _auto_cast(s: str) -> Any:
    """把字符串转成更合适的基础类型（int/float/bool/None），否则保持 str。"""
    low = s.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def _apply_cli_overrides_to_cfg(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """将解析后的 --set 覆盖写入到 yaml dict。"""
    for k, v in overrides.items():
        _set_by_path(cfg, k, v)


# =========================================================
# 权重加载 / 保存相关
# =========================================================
def load_model_pt(model: Any, path: Optional[str] = None, strict: bool = True) -> None:
    """
    加载权重：兼容 state_dict（字典）与整模型（序列化 Module）两种格式。

    参数：
      model: 已构建的模型实例
      path:  权重文件路径
      strict: 是否严格匹配 key（默认 True）
    """

    # 检测文件是否存在
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Weights file is not found or null.")

    # 加载权重文件
    ckpt = torch.load(path, map_location="cpu")

    # 兼容方式
    if isinstance(ckpt, dict) and ("model" in ckpt or "state_dict" in ckpt):
        sd = ckpt.get("model", ckpt.get("state_dict"))
        model.load_state_dict(sd, strict=strict)
    elif isinstance(ckpt, dict):
        # 可能是“裸 state_dict”
        model.load_state_dict(ckpt, strict=strict)
    else:
        raise ValueError("ckpt file is failed to load.")

    print(f"=> 已加载权重: {path}")


def _derive_resume_path(cfg: Dict[str, Any], trainer_args: Dict[str, Any]) -> Optional[str]:
    """
    从配置推导 resume 路径：
      - 若 cfg['resume'] 是 True，则尝试 save_dir/last.pt
      - 若 cfg['resume'] 是字符串，则直接使用该路径
    """
    resume = cfg.get("resume", False)
    if not resume:
        return None
    if isinstance(resume, str) and resume.strip():
        return resume
    save_dir = trainer_args.get("save_dir", "outputs/default_run")
    last = os.path.join(save_dir, "last.pt")
    return last if os.path.isfile(last) else None


def save_last_and_best(trainer: Any, save_dir: str) -> None:
    """
    训练结束后的统一收尾：
      - 始终保存 last.pt（若 BaseTrainer 已在内部保存，这里只做兜底）
      - 若能拿到“最佳”状态，则尝试保存/更新 best.pt
    """
    os.makedirs(save_dir, exist_ok=True)
    last_path = os.path.join(save_dir, "last.pt")
    try:
        # 若训练器提供了 state_dict 或模型/优化器等信息，自行组织
        torch.save({"model": getattr(trainer, "model", None).state_dict()
                    if hasattr(trainer, "model") else None}, last_path)
        print(f"=> 已保存 last.pt 到: {last_path}")
    except Exception as e:
        print(f"[warn] 保存 last.pt 失败: {e}")

    # 简单的 best 兜底：若存在 best 属性，复制一份
    best_path = os.path.join(save_dir, "best.pt")
    if os.path.isfile(best_path):
        return
    try:
        # 这里不强制保存 best（多数训练器会在更优时覆盖 best.pt）
        pass
    except Exception as e:
        print(f"[warn] 保存 best.pt 失败: {e}")


# =========================================================
# 随机性与环境
# =========================================================
def set_seed_and_deterministic(seed: Optional[int], deterministic: bool) -> None:
    """
    设置全局随机种子；在 deterministic=True 时开启确定性算法（速度可能变慢）。
    影响范围：random / numpy / torch（CPU/GPU）
    """
    import random
    import numpy as np
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # 更严格的确定性（可能会让性能下降）
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


# =========================================================
# 命令：训练 / 评估 / 导出
# =========================================================
def cmd_train(cfg: Dict[str, Any]) -> None:
    """
    训练主入口（从 YAML 配置启动一次完整训练）。

    改动点：先集中提取会用到的全部 YAML 参数（避免“黑箱”），
    再逐步执行构建模型/训练器、加载权重、构建数据、启动训练。
    """
    # -------------------------
    # 1) 从 YAML 提取所有参数
    # -------------------------
    # 顶层通用
    seed: Optional[int] = cfg.get("seed")
    deterministic: bool = bool(cfg.get("deterministic", False))
    verbose: bool = bool(cfg.get("verbose", True))
    resume_flag = cfg.get("resume", False)     # True / False / str 路径
    weights_path = cfg.get("weights", None)    # 初始权重（不是断点）

    # 模型段
    model_cfg: Dict[str, Any] = cfg.get("model", {}) or {}
    model_class = model_cfg.get("class")
    model_args: Dict[str, Any] = model_cfg.get("args", {}) or {}

    # 训练器段
    trainer_cfg: Dict[str, Any] = cfg.get("trainer", {}) or {}
    trainer_class = trainer_cfg.get("class")
    trainer_args: Dict[str, Any] = trainer_cfg.get("args", {}) or {}

    # 常用训练参数（从 trainer.args 中拎出来，便于代码阅读）
    epochs: int = int(trainer_args.get("epochs", 1))
    save_dir: str = trainer_args.get("save_dir", "outputs/default_run")
    img_size: int = int(trainer_args.get("img_size", 320))

    # 优化器/调度/训练技巧（仅提取常见字段，剩余保持透传给 Trainer）
    optimizer_cfg: Dict[str, Any] = trainer_args.get("optimizer_cfg", {}) or {}
    scheduler_name: str = trainer_args.get("scheduler_name", "MultiStepLR")
    milestones = trainer_args.get("milestones", [120, 200, 260])
    gamma: float = float(trainer_args.get("gamma", 0.1))
    use_amp: bool = bool(trainer_args.get("use_amp", True))
    use_ema: bool = bool(trainer_args.get("use_ema", True))
    ema_decay: float = float(trainer_args.get("ema_decay", 0.9998))
    clip_grad_norm: float = float(trainer_args.get("clip_grad_norm", 0.0))
    log_interval: int = int(trainer_args.get("log_interval", 50))

    # 数据段（train/val 两个 dataloader）
    data_cfg: Dict[str, Any] = cfg.get("data", {}) or {}
    train_loader_cfg: Dict[str, Any] = data_cfg.get("train", {}) or {}
    train_builder = train_loader_cfg.get("builder")
    train_args: Dict[str, Any] = train_loader_cfg.get("args", {}) or {}

    val_loader_cfg: Dict[str, Any] = data_cfg.get("val", {}) or {}
    val_builder = val_loader_cfg.get("builder")
    val_args: Dict[str, Any] = val_loader_cfg.get("args", {}) or {}

    # 由 resume_flag + save_dir 推导 resume 路径（若 resume_flag=True 则找 last.pt）
    resume_path = _derive_resume_path(cfg, trainer_args)

    if verbose:
        print("[cmd_train] 提取的关键参数：")
        print("  seed/deterministic:", seed, deterministic)
        print("  resume/weights:", resume_flag, weights_path)
        print("  model.class / args.keys:", model_class, list(model_args.keys()))
        print("  trainer.class / args.keys:", trainer_class, list(trainer_args.keys()))
        print("  epochs/save_dir/img_size:", epochs, save_dir, img_size)
        print("  optimizer_cfg:", optimizer_cfg)
        print("  scheduler_name/milestones/gamma:", scheduler_name, milestones, gamma)
        print("  use_amp/use_ema/ema_decay:", use_amp, use_ema, ema_decay)
        print("  clip_grad_norm/log_interval:", clip_grad_norm, log_interval)
        print("  train_loader.builder / args.keys:", train_builder, list(train_args.keys()))
        print("  val_loader.builder   / args.keys:", val_builder,   list(val_args.keys()))

    # -------------------------
    # 2) 设置随机性
    # -------------------------
    set_seed_and_deterministic(seed, deterministic)

    # -------------------------
    # 3) 构建模型与训练器（显式使用 model_class / model_args）
    # -------------------------
    if not model_class:
        raise ValueError("model.class 为空，请在 YAML 中配置 model.class")
    model_ctor = resolve_callable(model_class)
    model = maybe_instantiate(model_ctor, model_args)

    if not trainer_class:
        raise ValueError("trainer.class 为空，请在 YAML 中配置 trainer.class")
    train_ctor = resolve_callable(trainer_class)
    trainer = maybe_instantiate(train_ctor, {**trainer_args, "model": model})

    # -------------------------
    # 4) 权重 / 断点恢复（resume 优先于 weights）
    # -------------------------
    if resume_path and os.path.isfile(resume_path):
        if hasattr(trainer, "resume"):
            print(f"=> 从断点恢复: {resume_path}")
            trainer.resume(resume_path)
        else:
            print(f"=> 加载模型权重(断点): {resume_path}")
            load_model_pt(model, resume_path, strict=False)
    elif weights_path:
        print(f"=> 加载模型初始权重: {weights_path}")
        load_model_pt(model, weights_path, strict=False)

    # -------------------------
    # 5) 构建数据（train / val），显式使用 builder / args
    # -------------------------
    train_loader = None
    if train_loader_cfg:
        if not train_builder:
            raise ValueError("data.train.builder 为空")
        train_builder = resolve_callable(train_builder)
        train_loader = maybe_instantiate(train_builder, train_args)

    val_loader = None
    if val_loader_cfg:
        if not val_builder:
            raise ValueError("data.val.builder 为空")
        val_builder = resolve_callable(val_builder)
        val_loader = maybe_instantiate(val_builder, val_args)

    # -------------------------
    # 6) 启动训练（兼容不同签名）
    # -------------------------
    if hasattr(trainer, "train"):
        try:
            trainer.train(train_loader, val_loader)  # 标准签名：两个 loader
        except TypeError:
            trainer.train(train_loader)              # 兼容只收一个 loader 的训练器
    else:
        raise AttributeError("训练器缺少 train() 方法")

    # -------------------------
    # 7) 收尾保存（兜底；大多数 BaseTrainer 内部已有 best/last 保存）
    # -------------------------
    save_last_and_best(trainer, save_dir)


def cmd_eval(cfg: Dict[str, Any]) -> None:
    """
    评估/验证主入口。

    改动点：先集中提取会用到的全部 YAML 参数，再执行构建与评估。
    """
    # -------------------------
    # 1) 从 YAML 提取所有参数
    # -------------------------
    seed: Optional[int] = cfg.get("seed")
    deterministic: bool = bool(cfg.get("deterministic", False))
    verbose: bool = bool(cfg.get("verbose", True))

    # 模型段
    model_cfg: Dict[str, Any] = cfg.get("model", {}) or {}
    model_class = model_cfg.get("class")
    model_args: Dict[str, Any] = model_cfg.get("args", {}) or {}

    # 训练器段（评估时也需要构建训练器，用于 evaluate 逻辑和设备管理）
    trainer_cfg: Dict[str, Any] = cfg.get("trainer", {}) or {}
    trainer_class = trainer_cfg.get("class")
    trainer_args: Dict[str, Any] = trainer_cfg.get("args", {}) or {}
    save_dir: str = trainer_args.get("save_dir", "outputs/default_run")

    # 权重优先顺序：CLI/YAML 的 weights -> save_dir/{best.pt,last.pt}
    weights_path = cfg.get("weights", None)
    if not weights_path:
        best = os.path.join(save_dir, "best.pt")
        last = os.path.join(save_dir, "last.pt")
        weights_path = best if os.path.isfile(best) else (last if os.path.isfile(last) else None)

    # 数据段（只需要 val）
    data_cfg: Dict[str, Any] = cfg.get("data", {}) or {}
    val_loader_cfg: Dict[str, Any] = data_cfg.get("val", {}) or {}
    val_builder = val_loader_cfg.get("builder")
    val_args: Dict[str, Any] = val_loader_cfg.get("args", {}) or {}

    if verbose:
        print("[cmd_eval] 提取的关键参数：")
        print("  seed/deterministic:", seed, deterministic)
        print("  model.class / args.keys:", model_class, list(model_args.keys()))
        print("  trainer.class / args.keys:", trainer_class, list(trainer_args.keys()))
        print("  save_dir/weights_path:", save_dir, weights_path)
        print("  val_loader.builder / args.keys:", val_builder, list(val_args.keys()))

    # -------------------------
    # 2) 设置随机性
    # -------------------------
    set_seed_and_deterministic(seed, deterministic)

    # -------------------------
    # 3) 构建模型与训练器（显式使用 model_class / model_args）
    # -------------------------
    if not model_class:
        raise ValueError("model.class 为空，请在 YAML 中配置 model.class")
    model_ctor = resolve_callable(model_class)
    model = maybe_instantiate(model_ctor, model_args)

    if not trainer_class:
        raise ValueError("trainer.class 为空，请在 YAML 中配置 trainer.class")
    train_ctor = resolve_callable(trainer_class)
    trainer = maybe_instantiate(train_ctor, {**trainer_args, "model": model})

    # -------------------------
    # 4) 加载权重（若未提供则随机权重评估，通常仅用于烟测）
    # -------------------------
    if weights_path:
        load_model_pt(model, weights_path, strict=False)
    else:
        print("[warn] 未提供可用权重，将在随机初始化权重上评估（仅用于烟测）")

    # -------------------------
    # 5) 构建 val_loader（显式使用 builder / args）
    # -------------------------
    if not val_loader_cfg:
        raise ValueError("评估需要提供 data.val 段配置")
    if not val_builder:
        raise ValueError("data.val.builder 为空")
    val_builder = resolve_callable(val_builder)
    val_loader = maybe_instantiate(val_builder, val_args)

    # -------------------------
    # 6) 执行评估或前向烟测
    # -------------------------
    if hasattr(trainer, "evaluate"):
        metrics = trainer.evaluate(val_loader)
        print("==> Eval metrics:", metrics)
    else:
        print("[warn] 训练器不支持 evaluate()，仅做一次前向检查")
        batch = next(iter(val_loader))
        if isinstance(batch, (tuple, list)) and hasattr(trainer, "_move_batch_to_device"):
            batch = trainer._move_batch_to_device(batch)  # type: ignore
            model.eval()
            with torch.no_grad():
                out = model(batch[0])  # 假设第一个是 imgs
                print("forward ok; keys:", list(out) if isinstance(out, dict) else type(out))


def cmd_export(cfg: Dict[str, Any]) -> None:
    """
    模型导出主入口（ONNX / TorchScript）。

    改动点：保持原有“集中提取参数”的风格，并补充 wrapper/half/dynamic 等提取字段的注释。
    """
    # -------------------------
    # 1) 从 YAML 提取导出所需的全部参数
    # -------------------------
    exp_cfg = cfg.get("export", {}) or {}
    fmt: str = (exp_cfg.get("format") or "onnx").lower()      # onnx / torchscript(ts)
    output: str = exp_cfg.get("output") or "model.onnx"
    input_shape = exp_cfg.get("input", {}).get("shape") or [1, 3, 320, 320]
    opset: int = int(exp_cfg.get("opset", 13))
    use_half: bool = bool(exp_cfg.get("half", False))
    dynamic_batch: bool = bool(exp_cfg.get("dynamic_batch", False))
    verbose: bool = bool(cfg.get("verbose", True))

    # 可选 wrapper：对模型输入/输出做适配
    wrapper_cfg: Dict[str, Any] = exp_cfg.get("wrapper", {}) or {}
    wrapper_class = wrapper_cfg.get("class")
    wrapper_args: Dict[str, Any] = wrapper_cfg.get("args", {}) or {}

    # 模型段（导出同样需要按照 YAML 构建模型）
    model_cfg: Dict[str, Any] = cfg.get("model", {}) or {}
    model_class = model_cfg.get("class")
    model_args: Dict[str, Any] = model_cfg.get("args", {}) or {}

    # 权重（CLI > YAML.export.weights > YAML.weights）
    weights_path = cfg.get("weights", None) or exp_cfg.get("weights", None)

    if verbose:
        print("[cmd_export] 提取的关键参数：")
        print("  format/output:", fmt, output)
        print("  input_shape/opset:", input_shape, opset)
        print("  use_half/dynamic_batch:", use_half, dynamic_batch)
        print("  wrapper.class / args.keys:", wrapper_class, list(wrapper_args.keys()) if wrapper_class else None)
        print("  model.class / args.keys:", model_class, list(model_args.keys()))
        print("  weights_path:", weights_path)

    # -------------------------
    # 2) 构建模型并加载权重（显式使用 model_class / model_args）
    # -------------------------
    if not model_class:
        raise ValueError("model.class 为空，请在 YAML 中配置 model.class")
    ModelCtor = resolve_callable(model_class)
    model = maybe_instantiate(ModelCtor, model_args)

    if weights_path:
        load_model_pt(model, weights_path, strict=False)
    model.eval()

    # -------------------------
    # 3) （可选）包一层 wrapper
    # -------------------------
    if wrapper_class:
        wrapper_target = resolve_callable(wrapper_class)
        model = instantiate_with_filtered_kwargs(wrapper_target, {"model": model, **wrapper_args})
        print("=> 已使用 wrapper 包裹模型:", wrapper_class)

    if use_half:
        model.half()

    # -------------------------
    # 4) 准备 dummy 输入
    # -------------------------
    b, c, h, w = input_shape
    dummy = torch.randn(b, c, h, w)
    if use_half:
        dummy = dummy.half()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    # -------------------------
    # 5) 执行导出
    # -------------------------
    if fmt == "onnx":
        dynamic_axes = {"input": {0: "batch"}} if dynamic_batch else None
        torch.onnx.export(
            model,
            (dummy,),   # 转成Tuple形式
            output,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes
        )
        print(f"=> 已导出 ONNX 到: {output}")

    elif fmt in ("ts", "torchscript"):
        traced = torch.jit.trace(model, dummy)
        traced.save(output)
        print(f"=> 已导出 TorchScript 到: {output}")

    else:
        raise ValueError(f"未知导出格式: {fmt!r}")


# =========================================================
# CLI 解析与入口
# =========================================================
def build_argparser() -> argparse.ArgumentParser:
    """构建命令行解析器：支持 train / eval / export 三个子命令。"""
    p = argparse.ArgumentParser("Sparrow CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # 通用选项
    def add_common(sp: argparse.ArgumentParser):
        sp.add_argument("-c", "--config", required=True, help="YAML 配置文件路径")
        sp.add_argument("--set", dest="overrides", nargs="*", default=[],
                        help="临时覆盖配置，格式：--set a.b.c=val x.y=2")
        sp.add_argument("--save-dir", default="", help="覆盖 trainer.args.save_dir")
        sp.add_argument("--weights", default="", help="权重文件路径（优先级高于 YAML 中的 weights）")
        sp.add_argument("--continue", dest="cont", action="store_true", help="从 save_dir/last.pt 断点继续")

    # train
    sp_train = sub.add_parser("train", help="训练")
    add_common(sp_train)

    # eval
    sp_eval = sub.add_parser("eval", help="评估/验证")
    add_common(sp_eval)

    # export
    sp_export = sub.add_parser("export", help="导出 ONNX/TorchScript")
    add_common(sp_export)
    sp_export.add_argument("--format", default="", choices=["", "onnx", "torchscript", "ts"],
                           help="导出格式（默认 onnx）")
    sp_export.add_argument("--output", default="", help="导出文件路径")
    sp_export.add_argument("--opset", type=int, default=13, help="ONNX opset（默认 13）")
    sp_export.add_argument("--input-shape", default="", help="输入形状，如 1,3,320,320")
    sp_export.add_argument("--half", action="store_true", help="半精度导出（谨慎使用）")
    sp_export.add_argument("--dynamic-batch", action="store_true", help="ONNX 动态 batch 维度")

    return p


def _apply_cli_to_cfg(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    将常见的 CLI 选项回写到 YAML 配置中（统一下游逻辑）。
    优先级：CLI > YAML
    """
    # 顶层覆盖
    if args.cont:
        cfg["resume"] = True
    if args.weights:
        cfg["weights"] = args.weights

    # 覆盖 save_dir
    if args.save_dir:
        trainer = cfg.setdefault("trainer", {})
        targs = trainer.setdefault("args", {})
        targs["save_dir"] = args.save_dir

    # 导出子命令的附加覆盖
    if args.cmd == "export":
        exp = cfg.setdefault("export", {})
        if args.format:
            exp["format"] = args.format
        if args.output:
            exp["output"] = args.output
        if args.opset:
            exp["opset"] = args.opset
        if args.input_shape:
            # 解析形状 "1,3,320,320"
            try:
                shape = [int(x) for x in args.input_shape.strip().split(",")]
                exp.setdefault("input", {})["shape"] = shape
            except Exception:
                raise ValueError("--input-shape 需要逗号分隔的整数，如 1,3,320,320")
        if args.half:
            exp["half"] = True
        if args.dynamic_batch:
            exp["dynamic_batch"] = True

    # --set 覆盖
    overrides = _parse_cli_overrides(args.overrides)
    _apply_cli_overrides_to_cfg(cfg, overrides)


def main() -> None:
    """程序入口：解析命令行 -> 读取 YAML -> 应用覆盖 -> 执行子命令。"""
    parser = build_argparser()
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # 将 CLI 的覆盖项应用到 cfg
    _apply_cli_to_cfg(cfg, args)

    # 分发子命令
    if args.cmd == "train":
        cmd_train(cfg)
    elif args.cmd == "eval":
        cmd_eval(cfg)
    elif args.cmd == "export":
        cmd_export(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
