from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

from src.utils.paths import make_run_dir
from src.utils.project import find_project_root, resolve_project_path

DEFAULTS: dict[str, Any] = {
    "weights": "runs/detect/yolo12n_thermal_uav/weights/best.pt",
    "data": "configs/dataset_thermal_uav.yaml",
    "split": "val",
    "imgsz": 960,
    "batch": 16,
    "device": "0",
    "workers": 8,
    "project": "runs/val",
    "name": "thermal_uav_val",
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
    parser = argparse.ArgumentParser(description="Validate a trained YOLO thermal UAV detector on single or multi-UAV data.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained detector weights.")
    parser.add_argument("--data", type=str, default=None, help="Dataset YAML path.")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"], help="Split to evaluate.")
    parser.add_argument("--imgsz", type=int, default=None, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=None, help="Validation batch size.")
    parser.add_argument("--device", type=str, default=None, help='Device string: "0", "0,1", "cpu".')
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--project", type=str, default=None, help="Output project directory.")
    parser.add_argument("--name", type=str, default=None, help="Run name.")
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

    for key, value in vars(args).items():
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


def resolve_validation_config(cfg: dict[str, Any], project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or find_project_root()
    resolved = cfg.copy()
    resolved["weights"] = str(resolve_project_path(resolved["weights"], root, must_exist=True))
    resolved["data"] = str(resolve_project_path(resolved["data"], root, must_exist=True))
    resolved["project"] = str(resolve_project_path(resolved["project"], root, must_exist=False))
    resolved["device"] = resolve_device(resolved.get("device"))
    return resolved


def to_jsonable_metrics(metrics: Any) -> dict[str, Any]:
    results_dict = getattr(metrics, "results_dict", {}) or {}
    jsonable = {}
    for k, v in results_dict.items():
        if isinstance(v, (int, float)):
            jsonable[k] = float(v)
        else:
            try:
                jsonable[k] = float(v)
            except (TypeError, ValueError):
                jsonable[k] = str(v)
    return jsonable


def run_validation(cfg: dict[str, Any], project_root: Path | None = None) -> Path:
    resolved_cfg = resolve_validation_config(cfg, project_root=project_root)
    save_dir = make_run_dir(resolved_cfg["project"], resolved_cfg["name"], exist_ok=resolved_cfg["exist_ok"])

    print("Validation config:")
    for key in sorted(resolved_cfg.keys()):
        print(f"  {key}: {resolved_cfg[key]}")
    print(f"  save_dir: {save_dir}")

    model = YOLO(str(resolved_cfg["weights"]))
    metrics = model.val(
        data=resolved_cfg["data"],
        split=resolved_cfg["split"],
        imgsz=resolved_cfg["imgsz"],
        batch=resolved_cfg["batch"],
        device=resolved_cfg["device"],
        workers=resolved_cfg["workers"],
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        plots=True,
    )

    metrics_json = to_jsonable_metrics(metrics)
    metrics_file = save_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    print("\nValidation metrics:")
    if metrics_json:
        for k, v in metrics_json.items():
            print(f"  {k}: {v}")
    else:
        print("  Metrics dictionary is empty. Check Ultralytics logs in run directory.")
    print(f"\n[OK] Metrics saved to: {metrics_file.resolve()}")
    return metrics_file


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    run_validation(cfg)


if __name__ == "__main__":
    main()
