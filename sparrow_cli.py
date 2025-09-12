#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparrow_cli.py
Generic, YAML-driven CLI for training, evaluation, and export.

What's new in this revision:
- Robust .pt loading for both train/eval (resume if allowed; fallbacks if not specified).
- Standardized checkpoint saving: always writes last.pt; tries to produce best.pt.
- Fixed aliases to match repository layout.
- Optional reproducibility utilities (seed & deterministic mode).

Usage:
  python sparrow_cli.py train  -c configs/movenet.yaml
  python sparrow_cli.py eval   -c configs/movenet.yaml --set weights=outputs/mn/best.pt
  python sparrow_cli.py export -c configs/movenet.yaml --set export.output=out/mn.onnx
"""

import argparse
import importlib
import importlib.util
import os
import sys
import types
import shutil
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional, Union

import yaml
import numpy as np

# Optional torch imports are only required for train/eval/export
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None  # lazy guard when only inspecting configs
    nn = None
    optim = None


# -----------------------------
# Aliases (match your repo)
# -----------------------------

ALIASES: Dict[str, str] = {
    # Models
    "movenet": "sparrow.models.movenet.MoveNet",
    "ssdlite": "sparrow.models.ssdlite.SSDLite",
    # ONNX/TorchScript wrappers
    "dummy_movenet": "sparrow.models.onnx.dummy_movenet.DummyMoveNet",
    "dummy_ssdlite": "sparrow.models.onnx.dummy_ssdlite.DummySSDLite",
    # Datasets / Dataloaders
    "coco_kpts_dataloader": "sparrow.datasets.coco_kpts.create_kpts_dataloader",
    "simple_image_folder": "sparrow.datasets.simple_loader.SimpleImageFolder",
    # Trainers
    "kpts_trainer": "sparrow.task.kpts_trainer.KptsTrainer",
    "det_trainer": "sparrow.task.det_trainer.DetTrainer",
}


# =========================
# Reproducibility helpers
# =========================
def seed_everything(seed: int) -> None:
    """Fast mode (keeps cuDNN benchmark)."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass


def seed_everything_strict(seed: int) -> None:
    """Strict determinism (slower)."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass


# =========================
# Checkpoint I/O (.pt)
# =========================
PathLike = Union[str, os.PathLike]

def save_model_pt(
    model: "nn.Module",
    path: PathLike,
    optimizer: Optional["optim.Optimizer"] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    save_state_dict: bool = True,
) -> None:
    """Save checkpoint to .pt (recommended: state_dict-style)."""
    if torch is None:
        raise RuntimeError("PyTorch is required for saving checkpoints")
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
        torch.save(
            {
                "model_full": model,
                "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                "epoch": epoch,
                "extra": extra or {},
                "torch_version": torch.__version__,
            },
            path,
        )


def load_model_pt(
    path: PathLike,
    model: Optional["nn.Module"] = None,
    optimizer: Optional["optim.Optimizer"] = None,
    map_location: Union[str, "torch.device"] = "cpu",
    strict: bool = True,
) -> Tuple[Optional["nn.Module"], Dict[str, Any]]:
    """
    Load from .pt:
    - If checkpoint dict with 'model_state', load into provided model.
    - If checkpoint dict with 'model_full' and model is None, return that full model.
    - If bare state_dict (dict of tensors), require 'model' to be provided.
    - If a serialized nn.Module, return it if model is None else copy its state into provided model.
    Returns (model or None, meta).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for loading checkpoints")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {str(path)}")

    ckpt = torch.load(path, map_location=map_location)
    meta = {"epoch": None, "extra": {}, "torch_version": None}

    if isinstance(ckpt, dict):
        state = ckpt.get("model_state") or ckpt.get("state_dict")
        meta["epoch"] = ckpt.get("epoch")
        meta["extra"] = ckpt.get("extra", {})
        meta["torch_version"] = ckpt.get("torch_version")
        if model is None and "model_full" in ckpt:
            model = ckpt["model_full"]
        if state is not None and model is not None:
            # load_state_dict may return (missing, unexpected) on older PyTorch
            try:
                result = model.load_state_dict(state, strict=strict)
                if isinstance(result, tuple):
                    missing, unexpected = result
                    if missing or unexpected:
                        print(f"[load_model_pt] missing={missing}, unexpected={unexpected}")
            except Exception as e:
                print(f"[load_model_pt] Warning - load_state_dict failed: {e}")
        if optimizer is not None and ckpt.get("optimizer_state") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                print(f"[load_model_pt] Warning - failed to load optimizer: {e}")
        return model, meta

    if isinstance(ckpt, (dict,)):
        if model is None:
            raise ValueError("Provide a model to load a bare state_dict.")
        model.load_state_dict(ckpt, strict=strict)
        return model, meta

    if nn is not None and isinstance(ckpt, nn.Module):
        if model is None:
            model = ckpt
        else:
            model.load_state_dict(ckpt.state_dict(), strict=strict)
        return model, meta

    raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")


def atomic_copy(src: PathLike, dst: PathLike) -> None:
    """Copy file with a temp name then rename (best effort)."""
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def save_last_and_best(
    model: "nn.Module",
    trainer: Any,
    save_dir: PathLike,
    optimizer: Optional["optim.Optimizer"] = None,
    epoch: Optional[int] = None,
    best_state_dict: Optional[Dict[str, Any]] = None,
    best_metric: Optional[float] = None,
) -> None:
    """Always write last.pt; try best.pt if possible, else fall back to last."""
    save_dir = Path(save_dir)
    last_path = save_dir / "last.pt"
    best_path = save_dir / "best.pt"

    # Save last
    save_model_pt(model, last_path, optimizer=optimizer, epoch=epoch, extra={"best_metric": best_metric})

    # Determine best to save
    if best_state_dict is None:
        # Try common trainer attributes
        for attr in ("best_state_dict", "best_model_state", "best_model"):
            if hasattr(trainer, attr):
                val = getattr(trainer, attr)
                if isinstance(val, dict):
                    best_state_dict = val
                    break
                if nn is not None and isinstance(val, nn.Module):
                    best_state_dict = val.state_dict()
                    break

    if best_state_dict is not None:
        torch.save({"model_state": best_state_dict, "extra": {"best_metric": best_metric}}, best_path)
    else:
        # If trainer already produced best.pt, respect it; otherwise copy last -> best
        if not best_path.exists():
            try:
                atomic_copy(last_path, best_path)
            except Exception as e:
                print(f"[save_last_and_best] Warning - fallback copy failed: {e}")


# -----------------------------
# YAML utilities
# -----------------------------

def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_scalar(s: str) -> Any:
    """Try to parse scalars from string (bool/int/float/None); fallback to str."""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        if s.strip().startswith(("0x", "0X")):
            return int(s, 16)
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _set_by_dots(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expect key=value): {item}")
        key, val = item.split("=", 1)
        _set_by_dots(cfg, key, _parse_scalar(val))
    return cfg


# -----------------------------
# Dynamic import helpers
# -----------------------------

def _import_from_file(uri: str) -> Tuple[types.ModuleType, str]:
    """
    Load a module from file URI of form: file:/abs/path/to/mod.py#Attr
    Returns (module, attr_name).
    """
    if not uri.startswith("file:"):
        raise ValueError("file URI must start with 'file:'")
    body = uri[len("file:"):]
    if "#" not in body:
        raise ValueError("file URI must include '#ClassOrFunc', e.g., file:/path/x.py#Klass")
    file_path, attr = body.split("#", 1)
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Module file not found: {file_path}")
    mod_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module, attr


def resolve_callable(name_or_path: str) -> Callable[..., Any]:
    """
    Resolve a class or function from:
      - alias (ALIASES mapping)
      - dotted import path: "pkg.mod:Attr" or "pkg.mod.Attr"
      - file URI: "file:/abs/path/to/mod.py#Attr"
    Returns the attribute (class/function).
    """
    dotted = ALIASES.get(name_or_path, name_or_path)

    if dotted.startswith("file:"):
        module, attr = _import_from_file(dotted)
        if not hasattr(module, attr):
            raise AttributeError(f"{dotted} has no attribute '{attr}'")
        return getattr(module, attr)

    if ":" in dotted:
        mod_path, attr = dotted.split(":", 1)
    else:
        parts = dotted.split(".")
        if len(parts) < 2:
            raise ImportError(f"Invalid dotted path (need module.attr): {dotted}")
        mod_path, attr = ".".join(parts[:-1]), parts[-1]

    module = importlib.import_module(mod_path)
    if not hasattr(module, attr):
        raise AttributeError(f"{mod_path} has no attribute '{attr}'")
    return getattr(module, attr)


def maybe_instantiate(target: Any, kwargs: Dict[str, Any]) -> Any:
    """Class -> instance; callable -> call; else passthrough."""
    if isinstance(target, type):
        return target(**kwargs)
    if callable(target):
        return target(**kwargs)
    return target


# -----------------------------
# Builders
# -----------------------------

def build_model(cfg: Dict[str, Any]) -> Any:
    name = cfg.get("class")
    if not name:
        raise ValueError("model.class is required")
    cls = resolve_callable(name)
    args = cfg.get("args", {}) or {}
    return maybe_instantiate(cls, args)


def build_trainer(cfg: Dict[str, Any], model: Any) -> Any:
    name = cfg.get("class")
    if not name:
        raise ValueError("trainer.class is required")
    cls = resolve_callable(name)
    args = cfg.get("args", {}) or {}
    if "model" not in args:
        args["model"] = model
    return maybe_instantiate(cls, args)


def build_loader(section: Dict[str, Any]) -> Any:
    name = section.get("builder") or section.get("class")
    if not name:
        raise ValueError("data.<train|val>.builder (or class) is required")
    fn = resolve_callable(name)
    args = section.get("args", {}) or {}
    return maybe_instantiate(fn, args)


# -----------------------------
# Checkpoint helpers
# -----------------------------

def _derive_resume_path(cfg: Dict[str, Any], save_dir: Optional[str]) -> Optional[str]:
    """
    Returns a path for resuming checkpoints under precedence:
    1) cfg['resume'] as a string path
    2) if cfg['resume'] is True and save_dir is set -> save_dir/last.pt
    3) None otherwise
    """
    resume_cfg = cfg.get("resume", None)
    if isinstance(resume_cfg, str) and resume_cfg:
        return resume_cfg
    if resume_cfg is True and save_dir:
        p = os.path.join(save_dir, "last.pt")
        if os.path.isfile(p):
            return p
    return None


# -----------------------------
# Commands
# -----------------------------

def cmd_train(cfg: Dict[str, Any]):
    if torch is None:
        raise RuntimeError("PyTorch is required for training")

    # Seed (optional)
    seed = cfg.get("seed", None)
    if isinstance(seed, int):
        if cfg.get("deterministic", False):
            seed_everything_strict(seed)
        else:
            seed_everything(seed)

    # Build model & trainer
    model = build_model(cfg["model"])
    trainer = build_trainer(cfg["trainer"], model)

    # Deduce save_dir if present
    save_dir = cfg.get("trainer", {}).get("args", {}).get("save_dir", None)

    # Resume or init weights
    resume_path = _derive_resume_path(cfg, save_dir)
    weights_path = cfg.get("weights", None)

    if resume_path and os.path.isfile(resume_path):
        print(f"[Train] Resuming from: {resume_path}")
        load_model_pt(resume_path, model=model, optimizer=getattr(trainer, "optimizer", None), strict=False)
    elif weights_path and os.path.isfile(weights_path):
        print(f"[Train] Loading weights: {weights_path}")
        load_model_pt(weights_path, model=model, strict=False)

    # Data
    train_loader = build_loader(cfg["data"]["train"])
    val_loader = build_loader(cfg["data"]["val"]) if cfg["data"].get("val") else None

    # Train
    try:
        if val_loader is not None:
            trainer.train(train_loader, val_loader)
        else:
            trainer.train(train_loader)
    except TypeError:
        if val_loader is not None:
            trainer.train(train_loader, val_loader=val_loader)
        else:
            trainer.train(train_loader)

    # After fit: ensure last.pt/best.pt exist
    try:
        epoch_attr = getattr(trainer, "epoch", None)
        metric_attr = getattr(trainer, "best_metric", None)
        save_last_and_best(
            model=model,
            trainer=trainer,
            save_dir=save_dir or "./outputs",
            optimizer=getattr(trainer, "optimizer", None),
            epoch=epoch_attr if isinstance(epoch_attr, int) else None,
            best_state_dict=None,  # will be discovered from trainer if present
            best_metric=float(metric_attr) if metric_attr is not None else None,
        )
        print(f"[Train] Checkpoints ensured at: {save_dir or './outputs'} (last.pt / best.pt)")
    except Exception as e:
        print(f"[Train] Warning - checkpoint finalize failed: {e}")


def cmd_eval(cfg: Dict[str, Any]):
    if torch is None:
        raise RuntimeError("PyTorch is required for evaluation")

    # Seed (optional)
    seed = cfg.get("seed", None)
    if isinstance(seed, int):
        if cfg.get("deterministic", False):
            seed_everything_strict(seed)
        else:
            seed_everything(seed)

    # Build model & trainer (some trainers own evaluation loops)
    model = build_model(cfg["model"])
    trainer = build_trainer(cfg["trainer"], model)

    # Determine weights path:
    weights = cfg.get("weights", None)
    if not weights:
        # Fallback: try save_dir/{best.pt,last.pt}
        save_dir = cfg.get("trainer", {}).get("args", {}).get("save_dir", None)
        if save_dir:
            cand_best = os.path.join(save_dir, "best.pt")
            cand_last = os.path.join(save_dir, "last.pt")
            if os.path.isfile(cand_best):
                weights = cand_best
            elif os.path.isfile(cand_last):
                weights = cand_last

    if not weights or not os.path.isfile(weights):
        raise ValueError("Please provide a valid 'weights' path in YAML or via --set weights=...")

    print(f"[Eval] Loading weights: {weights}")
    load_model_pt(weights, model=model, strict=False)

    # Build val loader
    if "data" not in cfg or "val" not in cfg["data"]:
        raise ValueError("YAML must provide data.val for evaluation")
    val_loader = build_loader(cfg["data"]["val"])

    # Run evaluation
    if hasattr(trainer, "evaluate"):
        metrics = trainer.evaluate(val_loader)
        print("[Eval] metrics:", metrics)
    else:
        model.eval()
        with torch.no_grad():
            for _batch in val_loader:
                pass
        print("[Eval] Done (no explicit metrics; trainer has no evaluate())")


def _to_torch_dtype(name: str):
    name = (name or "float32").lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def cmd_export(cfg: Dict[str, Any]):
    if torch is None:
        raise RuntimeError("PyTorch is required for export")

    exp = cfg.get("export") or {}
    fmt = (exp.get("format") or "onnx").lower()
    out_path = exp.get("output") or "model.onnx"
    input_cfg = exp.get("input") or {}
    input_shape = input_cfg.get("shape")
    if not input_shape:
        raise ValueError("export.input.shape is required, e.g. [1,3,192,192]")

    # Build model
    model = build_model(cfg["model"])

    # Load weights (if any)
    weights = cfg.get("weights")
    if weights and os.path.isfile(weights):
        print(f"[Export] Loading weights: {weights}")
        load_model_pt(weights, model=model, strict=False)

    # Optional wrapper (e.g., DummyMoveNet)
    wrapper_cfg = exp.get("wrapper")
    if wrapper_cfg:
        wrap_cls = resolve_callable(wrapper_cfg["class"])
        wrap_args = wrapper_cfg.get("args", {}) or {}
        model = maybe_instantiate(wrap_cls, {"movenet": model, **wrap_args})

    model.eval()

    # Prepare dummy input
    dtype = _to_torch_dtype(input_cfg.get("dtype"))
    x = torch.randn(*input_shape, dtype=dtype)
    if exp.get("half") or dtype == torch.float16:
        model = model.half()
        x = x.half()

    # Export
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)

    if fmt == "onnx":
        opset = int(exp.get("opset", 13))
        dynamic_axes = exp.get("dynamic_axes", None)
        torch.onnx.export(model, x, out_path,
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes=dynamic_axes,
                          do_constant_folding=True,
                          opset_version=opset)
        print(f"[Export] ONNX saved to: {out_path} (opset={opset})")

    elif fmt in ("torchscript", "ts"):
        if exp.get("script", False):
            ts = torch.jit.script(model)
        else:
            ts = torch.jit.trace(model, x)
        ts.save(out_path)
        print(f"[Export] TorchScript saved to: {out_path}")
    else:
        raise ValueError(f"Unsupported export format: {fmt}")


# -----------------------------
# CLI parser
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sparrow_cli",
                                description="Generic YAML-driven CLI for Sparrow models.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser):
        sp.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config file")
        sp.add_argument("--set", "-s", action="append", default=[],
                        help="Override config with key=value (dot notation), e.g., --set trainer.args.epochs=10")

    add_common(sub.add_parser("train", help="Run training"))
    add_common(sub.add_parser("eval",  help="Run evaluation on validation set"))
    add_common(sub.add_parser("export", help="Export model to ONNX/TorchScript"))

    return p


def main(argv: Optional[List[str]] = None):
    parser = build_argparser()
    args = parser.parse_args(argv)

    cfg = _read_yaml(args.config)
    cfg = apply_overrides(cfg, args.set or [])

    # Basic validations
    if "model" not in cfg or "trainer" not in cfg:
        raise ValueError("YAML must define 'model' and 'trainer' sections")
    if "data" not in cfg and args.command != "export":
        raise ValueError("YAML must define 'data' section for train/eval")

    if args.command == "train":
        cmd_train(cfg)
    elif args.command == "eval":
        cmd_eval(cfg)
    elif args.command == "export":
        cmd_export(cfg)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
