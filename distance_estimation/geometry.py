from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from distance_estimation.types import CameraMetadata


@dataclass(slots=True)
class GeometryEstimate:
    nominal_m: float | None
    min_m: float | None
    max_m: float | None
    notes: str | None = None


def build_camera_metadata(frame_width: int, frame_height: int, camera_cfg: dict[str, Any]) -> CameraMetadata:
    default_hfov = _to_optional_float(camera_cfg.get("default_hfov_deg"))
    default_vfov = _to_optional_float(camera_cfg.get("default_vfov_deg"))
    hfov = _to_optional_float(camera_cfg.get("hfov_deg")) or default_hfov
    vfov = _to_optional_float(camera_cfg.get("vfov_deg")) or default_vfov
    fx = _to_optional_float(camera_cfg.get("fx_px"))
    fy = _to_optional_float(camera_cfg.get("fy_px"))
    used_fallback = False

    if fx is None and hfov is not None:
        fx = focal_length_from_fov(frame_width, hfov)
        used_fallback = camera_cfg.get("hfov_deg") is None
    if fy is None and vfov is not None:
        fy = focal_length_from_fov(frame_height, vfov)
        used_fallback = used_fallback or camera_cfg.get("vfov_deg") is None

    if fx is None:
        hfov = hfov or 60.0
        fx = focal_length_from_fov(frame_width, hfov)
        used_fallback = True
    if fy is None:
        vfov = vfov or 45.0
        fy = focal_length_from_fov(frame_height, vfov)
        used_fallback = True

    return CameraMetadata(
        width_px=frame_width,
        height_px=frame_height,
        fx_px=fx,
        fy_px=fy,
        hfov_deg=hfov,
        vfov_deg=vfov,
        source=str(camera_cfg.get("name", "camera")),
        used_fallback_intrinsics=used_fallback,
    )


def focal_length_from_fov(size_px: int, fov_deg: float) -> float:
    fov_rad = math.radians(max(1e-6, float(fov_deg)))
    return float(size_px) / (2.0 * math.tan(fov_rad / 2.0))


def estimate_distance_from_bbox(
    bbox_width_px: float,
    bbox_height_px: float,
    camera: CameraMetadata,
    object_priors: dict[str, Any],
) -> GeometryEstimate:
    width_px = max(float(bbox_width_px), 1e-6)
    height_px = max(float(bbox_height_px), 1e-6)
    nominal_width = max(float(object_priors.get("nominal_width_m", 0.35)), 1e-6)
    nominal_height = max(float(object_priors.get("nominal_height_m", 0.12)), 1e-6)
    min_width = max(float(object_priors.get("min_width_m", nominal_width * 0.5)), 1e-6)
    max_width = max(float(object_priors.get("max_width_m", nominal_width * 2.0)), min_width)
    min_height = max(float(object_priors.get("min_height_m", nominal_height * 0.5)), 1e-6)
    max_height = max(float(object_priors.get("max_height_m", nominal_height * 2.0)), min_height)

    distance_width = camera.fx_px * nominal_width / width_px if width_px > 0 else None
    distance_height = camera.fy_px * nominal_height / height_px if height_px > 0 else None
    min_candidates = []
    max_candidates = []
    nominal_candidates = []

    if distance_width is not None:
        nominal_candidates.append(distance_width)
        min_candidates.append(camera.fx_px * min_width / width_px)
        max_candidates.append(camera.fx_px * max_width / width_px)
    if distance_height is not None:
        nominal_candidates.append(distance_height)
        min_candidates.append(camera.fy_px * min_height / height_px)
        max_candidates.append(camera.fy_px * max_height / height_px)

    if not nominal_candidates:
        return GeometryEstimate(None, None, None, notes="invalid_bbox")

    nominal = sum(nominal_candidates) / len(nominal_candidates)
    return GeometryEstimate(
        nominal_m=nominal,
        min_m=min(min_candidates),
        max_m=max(max_candidates),
        notes="fallback_camera" if camera.used_fallback_intrinsics else None,
    )


def classify_range_bin(distance_m: float | None, thresholds_m: list[float]) -> str:
    if distance_m is None:
        return "unknown"
    sorted_thresholds = sorted(float(v) for v in thresholds_m)
    if not sorted_thresholds:
        return "unknown"
    if distance_m <= sorted_thresholds[0]:
        return "close"
    if len(sorted_thresholds) == 1 or distance_m <= sorted_thresholds[1]:
        return "medium"
    return "distant"


def _to_optional_float(value: Any) -> float | None:
    if value in {None, "", "null"}:
        return None
    return float(value)
