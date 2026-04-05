from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CameraMetadata:
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    hfov_deg: float | None = None
    vfov_deg: float | None = None
    source: str = "unknown"
    used_fallback_intrinsics: bool = False


@dataclass(slots=True)
class DetectionInput:
    frame_index: int
    class_id: int
    score: float
    x_center: float
    y_center: float
    width: float
    height: float
    track_id: int | None = None
    label: str | None = None


@dataclass(slots=True)
class DepthPatchStats:
    median_m: float | None = None
    mean_m: float | None = None
    std_m: float | None = None
    valid_fraction: float = 0.0


@dataclass(slots=True)
class RangeEstimate:
    frame_index: int
    class_id: int
    score: float
    x_center: float
    y_center: float
    width: float
    height: float
    track_id: int | None = None
    distance_m: float | None = None
    distance_std_m: float | None = None
    distance_confidence: float | None = None
    range_bin: str = "unknown"
    low_confidence: bool = True
    geometric_distance_m: float | None = None
    depth_distance_m: float | None = None
    distance_min_m: float | None = None
    distance_max_m: float | None = None
    used_fallback_camera: bool = False
    notes: str | None = None

    def as_csv_row(self) -> dict[str, Any]:
        return {
            "frame": self.frame_index,
            "track_id": self.track_id if self.track_id is not None else "",
            "class_id": self.class_id,
            "x": round(self.x_center, 4),
            "y": round(self.y_center, 4),
            "w": round(self.width, 4),
            "h": round(self.height, 4),
            "score": round(self.score, 6),
            "distance_m": _round_or_blank(self.distance_m),
            "distance_std_m": _round_or_blank(self.distance_std_m),
            "distance_confidence": _round_or_blank(self.distance_confidence),
            "range_bin": self.range_bin,
            "low_confidence": self.low_confidence,
            "distance_min_m": _round_or_blank(self.distance_min_m),
            "distance_max_m": _round_or_blank(self.distance_max_m),
            "geometric_distance_m": _round_or_blank(self.geometric_distance_m),
            "depth_distance_m": _round_or_blank(self.depth_distance_m),
            "used_fallback_camera": self.used_fallback_camera,
            "notes": self.notes or "",
        }


@dataclass(slots=True)
class TrackRangeEstimate:
    frame_index: int
    track_id: int
    distance_m_raw: float
    distance_m_filtered: float
    distance_m_refined: float
    distance_std_m: float
    distance_confidence: float
    range_bin: str
    low_confidence: bool

    def as_csv_row(self) -> dict[str, Any]:
        return {
            "frame": self.frame_index,
            "track_id": self.track_id,
            "distance_m_raw": round(self.distance_m_raw, 4),
            "distance_m_filtered": round(self.distance_m_filtered, 4),
            "distance_m_refined": round(self.distance_m_refined, 4),
            "distance_std_m": round(self.distance_std_m, 4),
            "distance_confidence": round(self.distance_confidence, 6),
            "range_bin": self.range_bin,
            "low_confidence": self.low_confidence,
        }


def _round_or_blank(value: float | None) -> float | str:
    if value is None:
        return ""
    return round(float(value), 4)
