import os
from typing import Optional, Dict, Any, Tuple, Iterable, Set

import yaml
from sparrow.utils.logger import logger


# ---- 统一维护：BaseTrainer 会“占用”的关键字（避免与 **kwargs 冲突） ----
BASE_TRAINER_RESERVED_KEYS: Set[str] = {
    # 基础
    "data_dir", "save_dir", "device", "model", "loss_fn",
    "seed", "experiment_name", "resume",

    # dataloader 相关
    "batch_size", "num_workers", "pin_memory",

    # 训练轮次/混合精度/EMA/梯度裁剪
    "epochs", "use_amp", "use_ema", "ema_decay",
    "use_clip_grad", "clip_grad_norm",

    # 优化器
    "optimizer_name", "lr", "weight_decay",
    "betas", "eps", "amsgrad", "momentum", "nesterov",
    "alpha", "centered", "lr_decay", "rho",

    # 学习率调度器（尽量全）
    "scheduler_name", "T_max", "eta_min",
    "step_size", "gamma", "milestones",
    "mode", "factor", "patience", "threshold", "min_lr", "verbose",
    "max_lr", "steps_per_epoch", "pct_start", "anneal_strategy",
    "div_factor", "final_div_factor", "total_iters",
    "start_factor", "end_factor", "use_warmup_scheduler", "warmup_epochs",
    "power",
}

def _split_reserved(cfg: Dict[str, Any],
                    reserved_keys: Optional[Iterable[str]] = None
                    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    返回 (core, extra)
    - core: 仅包含保留键（通常由 BaseTrainer 显式接收）
    - extra: 其余键，安全地作为 **kwargs 继续下传
    """
    reserved = set(reserved_keys) if reserved_keys is not None else BASE_TRAINER_RESERVED_KEYS
    core = {k: v for k, v in cfg.items() if k in reserved}
    extra = {k: v for k, v in cfg.items() if k not in reserved}
    return core, extra


def _pretty_print_config(cfg: Dict[str, Any]) -> None:
    """
    结构化输出配置：
    - 顶层按键名排序
    - 若值是 dict，则作为一个小节打印子项（缩进 2 空格）
    - 其他标量键按表格列对齐：Key | Value | (Type)
    """
    def _format_kv(k: str, v: Any) -> Tuple[str, str, str]:
        v_disp = yaml.dump(v, default_flow_style=True).strip() if isinstance(v, (list, dict)) else str(v)
        ty = type(v).__name__
        return k, v_disp, ty

    scalar_items = []
    section_items = []

    for k in sorted(cfg.keys()):
        v = cfg[k]
        if isinstance(v, dict):
            section_items.append((k, v))
        else:
            scalar_items.append(_format_kv(k, v))

    # 计算列宽
    def _col_width(rows, idx):
        return max((len(r[idx]) for r in rows), default=0)

    key_w  = _col_width(scalar_items, 0)
    val_w  = _col_width(scalar_items, 1)

    print("\n[config] ========= Effective Configuration =========")
    if scalar_items:
        header = f"{'Key'.ljust(key_w)}  {'Value'.ljust(val_w)}  (Type)"
        print(header)
        print("-" * len(header))
        for k, v, ty in scalar_items:
            print(f"{k.ljust(key_w)}  {v.ljust(val_w)}  ({ty})")

    for sec_name, sec_dict in section_items:
        print(f"\n[{sec_name}]")
        if not sec_dict:
            print("  <empty>")
            continue
        # 子项按键名排序
        sub_items = []
        for sk in sorted(sec_dict.keys()):
            sub_items.append(_format_kv(sk, sec_dict[sk]))
        sk_w = max((len(i[0]) for i in sub_items), default=0)
        sv_w = max((len(i[1]) for i in sub_items), default=0)
        print(f"  {'Key'.ljust(sk_w)}  {'Value'.ljust(sv_w)}  (Type)")
        print("  " + "-" * (2 + sk_w + 2 + sv_w + 7))
        for sk, sv, sty in sub_items:
            print(f"  {sk.ljust(sk_w)}  {sv.ljust(sv_w)}  ({sty})")
    print("[config] ===========================================\n")


def update_from_yaml(yaml_path: Optional[str] = None,
                     default_dict: Optional[Dict[str, Any]] = None,
                     *,
                     reserved_keys: Optional[Iterable[str]] = None,
                     return_extra: bool = False
                     ) -> Any:
    """
    以 YAML 为主、默认兜底；可选返回 (cfg, extra_cfg)：
      - cfg: 完整配置（用于 self.cfg 及 get(...)）
      - extra_cfg: 剔除保留键后的剩余项，安全用于 **kwargs 继续传递

    用法：
        cfg, extra = update_from_yaml(path, default_dict, return_extra=True)
        # super(..., **extra)

    兼容老用法：
        cfg = update_from_yaml(path, default_dict)   # 仅返回 cfg
    """
    base = (default_dict or {}).copy()

    yaml_cfg: Dict[str, Any] = {}
    if yaml_path and isinstance(yaml_path, str) and os.path.isfile(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    yaml_cfg = loaded
                elif loaded is None:
                    logger.warning("[config] YAML is empty: %s -> using defaults.", yaml_path)
                else:
                    logger.warning("[config] YAML content not a dict (%s), using defaults.", type(loaded).__name__)
        except Exception as e:
            logger.warning("[config] Failed to load YAML: %s; Using defaults. Error: %s", yaml_path, e)
    elif yaml_path:
        logger.warning("[config] YAML not found: %s; Using defaults.", yaml_path)
    else:
        print("[config] No YAML path provided, using defaults")

    cfg: Dict[str, Any] = base
    cfg.update(yaml_cfg)

    if yaml_cfg:
        print(f"[config] Loaded configuration from: {yaml_path}")
        print(f"[config] Updated {len(yaml_cfg)} keys: {', '.join(sorted(yaml_cfg.keys()))}")
    else:
        print("[config] Using default configuration only")

    _pretty_print_config(cfg)

    if return_extra:
        _, extra = _split_reserved(cfg, reserved_keys)
        return cfg, extra
    return cfg