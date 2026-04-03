from __future__ import annotations

from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def resolve_existing_path(path_value: str | Path, description: str = "path") -> Path:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def get_video_fps(source: Path, fallback_fps: float = 30.0) -> float:
    if not is_video_file(source):
        return fallback_fps

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return fallback_fps

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 1e-6 else fallback_fps


def make_run_dir(project: str | Path, name: str, exist_ok: bool = False) -> Path:
    project_path = Path(project)
    base = project_path / name

    if exist_ok or not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base

    idx = 2
    while True:
        candidate = project_path / f"{name}{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        idx += 1
