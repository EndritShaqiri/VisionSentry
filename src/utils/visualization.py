from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_preview_images(frame_dir: str | Path, max_images: int = 4) -> list[np.ndarray]:
    frame_path = Path(frame_dir)
    if not frame_path.exists():
        return []

    image_files = sorted([p for p in frame_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    preview = []
    for image_file in image_files[:max_images]:
        image_bgr = cv2.imread(str(image_file))
        if image_bgr is None:
            continue
        preview.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return preview
