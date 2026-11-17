"""Environment detection shared by Kaggle, Colab and local CPU runs."""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvInfo:
    platform: str
    gpu: str | None
    cuda: str | None
    torch: str | None
    transformers: str | None
    triton: str | None
    python: str


def detect_platform() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if "google.colab" in sys.modules or os.environ.get("COLAB_GPU") is not None:
        return "colab"
    return "local"


def output_dir(platform_name: str) -> Path:
    if platform_name == "kaggle":
        return Path("/kaggle/working/miniserve")
    if platform_name == "colab":
        return Path("/content/drive/MyDrive/miniserve")
    return Path.cwd() / "bench" / "results"


def env_info() -> EnvInfo:
    try:
        import torch
    except ImportError:
        torch = None

    def version(name: str) -> str | None:
        try:
            module = __import__(name)
            return getattr(module, "__version__", None)
        except ImportError:
            return None

    gpu = None
    if torch is not None and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)

    return EnvInfo(
        platform=detect_platform(),
        gpu=gpu,
        cuda=getattr(torch.version, "cuda", None) if torch else None,
        torch=torch.__version__ if torch else None,
        transformers=version("transformers"),
        triton=version("triton"),
        python=platform.python_version(),
    )


def write_env_file(destination: Path | str | None = None) -> str:
    name = detect_platform()
    path = Path(destination) if destination else output_dir(name) / "env" / "versions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(env_info()), indent=2))
    return str(path)
