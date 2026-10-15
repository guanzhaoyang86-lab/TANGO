from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

# 统一随机种子，保证实验可复现。
def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# 读取 YAML 配置文件。
def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    if Path(path).suffix.lower() == ".json":
        return json.loads(raw_text)
    try:
        import yaml

        return yaml.safe_load(raw_text)
    except ImportError:
        return json.loads(raw_text)


# 创建目录。
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# 以 JSON 格式保存对象，便于后续查日志。
def dump_json(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# 以 YAML 格式保存对象，便于保留实际生效的训练配置快照。
def dump_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        try:
            import yaml

            yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)
        except ImportError:
            json.dump(obj, f, ensure_ascii=False, indent=2)


def get_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def get_world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def is_main_process() -> bool:
    return get_rank() == 0


def distributed_barrier() -> None:
    try:
        import torch.distributed as dist
    except ImportError:
        return

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
