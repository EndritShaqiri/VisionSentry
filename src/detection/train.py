from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

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


def str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv12 detector for thermal UAV detection.")
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
    parser.add_argument("--cache", type=str2bool, default=None, help="Cache images for faster training.")
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


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    data_yaml = Path(cfg["data"])
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}. Update configs/dataset_thermal_uav.yaml before training."
        )

    print("Training config:")
    for key in sorted(cfg.keys()):
        print(f"  {key}: {cfg[key]}")

    model = YOLO(cfg["model"])
    model.train(
        data=cfg["data"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        epochs=cfg["epochs"],
        device=cfg["device"],
        workers=cfg["workers"],
        project=cfg["project"],
        name=cfg["name"],
        pretrained=cfg["pretrained"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        patience=cfg["patience"],
        cache=cfg["cache"],
        exist_ok=cfg["exist_ok"],
    )

    save_dir = Path(getattr(model.trainer, "save_dir", Path(cfg["project"]) / cfg["name"]))
    print(f"\n[OK] Training finished. Artifacts saved in: {save_dir.resolve()}")
    print(f"[OK] Best checkpoint (expected): {(save_dir / 'weights' / 'best.pt').resolve()}")


if __name__ == "__main__":
    main()
