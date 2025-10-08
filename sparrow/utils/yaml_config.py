import os
from typing import Optional, Dict, Any

import yaml

from sparrow.utils.logger import logger


def update_from_yaml(yaml_path: Optional[str] = None,
                     default_dict: Optional[dict] = None) -> Dict[str, Any]:
    """
    从 YAML 文件更新配置，支持部分覆盖和类型验证

    功能：
      1. 如果 YAML 文件不存在或路径为空，使用默认配置
      2. 如果 YAML 内容为空或格式错误，使用默认配置并发出警告
      3. 只更新 YAML 中存在的键，其他保持默认值
      4. 对关键配置进行类型和范围检查

    参数:
      yaml_path: YAML 配置文件路径（None 或空字符串表示使用默认配置）
      default_dict: 默认配置

    返回:
      更新后的配置字典

    配置文件示例（config.yaml）:
      ```yaml
      # 训练配置
      data_root: "/data/biwi"
      epochs: 50
      batch_size: 32
      lr: 1e-4

      # 可选配置（不写则使用默认值）
      # workers: 4
      # use_amp: false
      ```

    使用示例:
      >>> # 场景 1：使用默认配置
      >>> cfg = update_from_yaml()
      >>>
      >>> # 场景 2：从 YAML 更新
      >>> cfg = update_from_yaml("configs/train.yaml")
      >>>
      >>> # 场景 3：文件不存在时的容错
      >>> cfg = update_from_yaml("non_existent.yaml")  # 返回默认配置
    """
    # 1. 复制默认配置（避免修改全局 CFG）
    if default_dict is None:
        config = {}
    else:
        config = default_dict.copy()

    # 2. 检查 YAML 路径是否有效
    if not yaml_path or not isinstance(yaml_path, str):
        print("[config] No YAML path provided, using default configuration")
        return config

    # 3. 检查文件是否存在
    if not os.path.isfile(yaml_path):
        logger.warning("update_from_yaml",
            f"[config] YAML file not found: {yaml_path}\n"
            f"         Using default configuration instead."
        )
        return config

    # 4. 尝试加载 YAML 文件
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)

        # 检查 YAML 内容是否为空
        if yaml_config is None:
            logger.warning("update_from_yaml",
                f"[config] YAML file is empty: {yaml_path}\n"
                f"         Using default configuration."
            )
            return config

        # 检查是否为字典类型
        if not isinstance(yaml_config, dict):
            logger.warning("update_from_yaml",
                f"[config] YAML content is not a dictionary: {yaml_path}\n"
                f"         Expected dict, got {type(yaml_config).__name__}\n"
                f"         Using default configuration."
            )
            return config

    except yaml.YAMLError as e:
        logger.warning("update_from_yaml",
            f"[config] Failed to parse YAML file: {yaml_path}\n"
            f"         Error: {e}\n"
            f"         Using default configuration."
        )
        return config
    except Exception as e:
        logger.warning("update_from_yaml",
            f"[config] Unexpected error loading YAML: {yaml_path}\n"
            f"         Error: {e}\n"
            f"         Using default configuration."
        )
        return config

    # 5. 合并配置（只更新 YAML 中存在的键）
    updated_keys = []
    invalid_keys = []

    for key, value in yaml_config.items():
        # 检查键是否在默认配置中
        if key not in config:
            invalid_keys.append(key)
            continue

        # 类型检查：确保新值类型与默认值兼容
        default_value = config[key]
        default_type = type(default_value)

        # 特殊处理：空字符串和 None 可以互换
        if default_value == "" and value is None:
            config[key] = ""
            updated_keys.append(key)
            continue

        # 特殊处理：数值类型的兼容（int 和 float）
        if isinstance(default_value, (int, float)) and isinstance(value, (int, float)):
            config[key] = type(default_value)(value)  # 转换为默认类型
            updated_keys.append(key)
            continue

        # 一般类型检查
        if not isinstance(value, default_type):
            logger.warning("update_from_yaml",
                f"[config] Type mismatch for key '{key}': "
                f"expected {default_type.__name__}, got {type(value).__name__}\n"
                f"         Skipping this key."
            )
            continue

        # 更新配置
        config[key] = value
        updated_keys.append(key)

    # 6. 打印配置更新摘要
    print(f"[config] Loaded configuration from: {yaml_path}")
    if updated_keys:
        print(f"[config] Updated {len(updated_keys)} keys: {', '.join(updated_keys)}")
    else:
        print("[config] No valid keys updated from YAML")

    if invalid_keys:
        logger.warning("update_from_yaml",
            f"[config] Found {len(invalid_keys)} invalid keys (not in default config): "
            f"{', '.join(invalid_keys)}\n"
            f"         These keys will be ignored."
        )

    return config


def save_config_to_yaml(config: Dict[str, Any], save_path: str) -> None:
    """
    将配置保存为 YAML 文件（方便复现实验）

    参数:
      config: 配置字典
      save_path: 保存路径

    示例:
      >>> cfg = update_from_yaml("config.yaml")
      >>> save_config_to_yaml(cfg, "runs/exp1/config_used.yaml")
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("save_config_to_yaml",
                f"[config] Configuration saved to: {save_path}")
