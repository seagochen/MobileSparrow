# core/utils/paths.py
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Iterable, List, Union

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

def iter_image_paths(src: Union[str, Path, Iterable[Union[str, Path]]],
                     recursive: bool = True,
                     exts: Iterable[str] = None) -> List[str]:
    """
    统一收集图片绝对路径：
      - src: 目录 / 单个文件 / 通配符('*.jpg') / 文本清单(.txt/.list，每行一个路径，支持注释#)
      - recursive: 目录时是否递归
      - exts: 允许扩展名集合（默认见 IMG_EXTS）
    返回：去重、排序的绝对路径列表
    """
    if exts is None:
        exts = IMG_EXTS
    exts = {e if e.startswith('.') else ('.' + e) for e in (x.lower() for x in exts)}

    def _gather_one(p: Union[str, Path]) -> List[str]:
        p = Path(str(p))
        if not str(p).strip():
            return []
        if p.is_dir():
            glb = "**/*" if recursive else "*"
            return [str(f.resolve()) for f in p.glob(glb) if f.suffix.lower() in exts]
        # 通配符
        if any(ch in str(p) for ch in "*?["):
            return [str(f.resolve()) for f in Path().glob(str(p)) if f.suffix.lower() in exts]
        # 普通文件
        if p.is_file():
            if p.suffix.lower() in exts:
                return [str(p.resolve())]
            if p.suffix.lower() in {".txt", ".list"}:
                out: List[str] = []
                for line in p.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    out += _gather_one(s)
                return out
        return []

    # 支持可迭代输入
    paths = []
    if isinstance(src, (list, tuple, set)):
        for it in src:
            paths += _gather_one(it)
    else:
        paths = _gather_one(src)

    # 去重 + 排序
    seen, uniq = set(), []
    for x in paths:
        if x not in seen:
            uniq.append(x); seen.add(x)
    uniq.sort()
    return uniq
