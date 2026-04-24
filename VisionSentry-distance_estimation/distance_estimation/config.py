from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.project import find_project_root, resolve_project_path

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RANGING_CONFIG = PACKAGE_ROOT / "configs" / "ranging.yaml"


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_ranging_bundle(
    *,
    config_path: str | Path | None = None,
    camera_config_path: str | Path | None = None,
    modality: str | None = None,
    project_root: Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    root = project_root or find_project_root()
    cfg = load_yaml_file(DEFAULT_RANGING_CONFIG)

    resolved_cfg_path = None
    if config_path is not None:
        resolved_cfg_path = resolve_project_path(config_path, root, must_exist=True)
        cfg = deep_merge(cfg, load_yaml_file(resolved_cfg_path))

    if modality is not None:
        cfg["modality"] = modality

    resolved_camera_path = None
    camera_path_value = camera_config_path or cfg.get("camera_profile_path")
    if camera_path_value:
        resolved_camera_path = resolve_project_path(camera_path_value, root, must_exist=True)
        camera_cfg = load_yaml_file(resolved_camera_path)
        cfg["camera"] = deep_merge(cfg.get("camera", {}), camera_cfg)
        cfg["camera_profile_path"] = str(resolved_camera_path)

    if resolved_cfg_path is not None:
        cfg["_config_path"] = str(resolved_cfg_path)
    return cfg, resolved_camera_path
