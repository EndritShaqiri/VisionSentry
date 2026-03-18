from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

from src.utils.project import find_project_root, resolve_project_path

DEFAULTS: dict[str, Any] = {
    "model": "yolo12n.pt",
    "data": "configs/dataset_thermal_uav.yaml",
    "imgsz": 960,
    "batch": 16,
    "epochs": 100,
    "device": "0",
    "workers": 8,
    "project": "runs/detect",
    "name": "yolo12n_thermal_uav",
    "pretrained": True,
    "optimizer": "auto",
    "lr0": 0.01,
    "patience": 30,
    "cache": False,
    "exist_ok": False,
}
OPTIONAL_TRAIN_ARGS = ("amp", "verbose", "seed", "deterministic")


def str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def str2cache(value: str) -> bool | str:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    if lowered in {"ram", "disk"}:
        return lowered
    raise argparse.ArgumentTypeError(f"Invalid cache value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv12 detector for single or multi-UAV thermal or RGB detection.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--model", type=str, default=None, help="Model weights or model YAML (e.g., yolo12n.pt).")
    parser.add_argument("--data", type=str, default=None, help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=None, help="Training image size.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default=None, help='Device string: "0", "0,1", "cpu".')
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--project", type=str, default=None, help="Output project directory.")
    parser.add_argument("--name", type=str, default=None, help="Run name.")
    parser.add_argument("--pretrained", type=str2bool, default=None, help="Use pretrained backbone.")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer choice (auto/SGD/Adam/AdamW).")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--cache", type=str2cache, default=None, help='Cache images (true/false/"ram"/"disk").')
    parser.add_argument("--amp", type=str2bool, default=None, help="Enable automatic mixed precision.")
    parser.add_argument("--verbose", type=str2bool, default=None, help="Enable verbose Ultralytics logging.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for training.")
    parser.add_argument("--deterministic", type=str2bool, default=None, help="Enable deterministic training mode.")
    parser.add_argument("--exist_ok", type=str2bool, default=None, help="Allow overwriting existing run folder.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = DEFAULTS.copy()
    if args.config:
        cfg.update(load_yaml(args.config))

    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if key == "config" or value is None:
            continue
        cfg[key] = value
    return cfg


def resolve_device(device: str | None) -> str:
    if device is None:
        return "0" if torch.cuda.is_available() else "cpu"

    lowered = str(device).strip().lower()
    if lowered == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return str(device)


def resolve_training_config(cfg: dict[str, Any], project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or find_project_root()
    resolved = cfg.copy()

    data_yaml = resolve_project_path(resolved["data"], root, must_exist=True)
    resolved["data"] = str(data_yaml)

    model_value = str(resolved["model"])
    model_path = Path(model_value)
    if model_path.is_absolute() or any(sep in model_value for sep in ("/", "\\")):
        resolved["model"] = str(resolve_project_path(model_path, root, must_exist=True))

    resolved["project"] = str(resolve_project_path(resolved["project"], root, must_exist=False))
    resolved["device"] = resolve_device(resolved.get("device"))
    return resolved


def run_training(cfg: dict[str, Any], project_root: Path | None = None) -> Path:
    resolved_cfg = resolve_training_config(cfg, project_root=project_root)

    print("Training config:")
    for key in sorted(resolved_cfg.keys()):
        print(f"  {key}: {resolved_cfg[key]}")

    train_kwargs: dict[str, Any] = {
        "data": resolved_cfg["data"],
        "imgsz": resolved_cfg["imgsz"],
        "batch": resolved_cfg["batch"],
        "epochs": resolved_cfg["epochs"],
        "device": resolved_cfg["device"],
        "workers": resolved_cfg["workers"],
        "project": resolved_cfg["project"],
        "name": resolved_cfg["name"],
        "pretrained": resolved_cfg["pretrained"],
        "optimizer": resolved_cfg["optimizer"],
        "lr0": resolved_cfg["lr0"],
        "patience": resolved_cfg["patience"],
        "cache": resolved_cfg["cache"],
        "exist_ok": resolved_cfg["exist_ok"],
    }
    for key in OPTIONAL_TRAIN_ARGS:
        value = resolved_cfg.get(key)
        if value is not None:
            train_kwargs[key] = value

    model = YOLO(resolved_cfg["model"])
    model.train(**train_kwargs)

    save_dir = Path(getattr(model.trainer, "save_dir", Path(resolved_cfg["project"]) / resolved_cfg["name"])).resolve()
    print(f"\n[OK] Training finished. Artifacts saved in: {save_dir}")
    print(f"[OK] Best checkpoint (expected): {(save_dir / 'weights' / 'best.pt').resolve()}")
    return save_dir


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    run_training(cfg)


if __name__ == "__main__":
    main()
