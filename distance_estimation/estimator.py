from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import torch

from distance_estimation.backbones import build_depth_backbone
from distance_estimation.config import load_ranging_bundle
from distance_estimation.geometry import build_camera_metadata, classify_range_bin, estimate_distance_from_bbox
from distance_estimation.models import DroneRangeHead
from distance_estimation.types import DepthPatchStats, DetectionInput, RangeEstimate


class RangeEstimator:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.depth_backbone = build_depth_backbone(cfg.get("depth_backbone", {}))
        self.range_head_name = str(cfg.get("range_head", {}).get("name", "heuristic")).lower()
        self.range_head = None
        self.feature_columns = list(cfg.get("feature_columns", DEFAULT_FEATURE_COLUMNS))
        self.runtime_notes: list[str] = []
        self._load_range_head_if_available()
        warning = getattr(self.depth_backbone, "warning", None)
        if warning:
            self.runtime_notes.append(str(warning))

    @classmethod
    def from_paths(
        cls,
        *,
        config_path: str | Path | None = None,
        camera_config_path: str | Path | None = None,
        modality: str | None = None,
    ) -> "RangeEstimator":
        cfg, _ = load_ranging_bundle(config_path=config_path, camera_config_path=camera_config_path, modality=modality)
        return cls(cfg)

    def save_runtime_config(self, save_dir: Path) -> Path:
        runtime_path = save_dir / "ranging_runtime.yaml"
        serializable_cfg = dict(self.cfg)
        serializable_cfg["runtime_notes"] = list(self.runtime_notes)
        with runtime_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(serializable_cfg, f, sort_keys=False)
        return runtime_path

    def estimate_detections(
        self,
        frame_bgr: np.ndarray,
        detections: list[DetectionInput],
        *,
        modality: str | None = None,
    ) -> list[RangeEstimate]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        modality_name = modality or str(self.cfg.get("modality", "ir"))
        frame_h, frame_w = frame_bgr.shape[:2]
        camera_metadata = build_camera_metadata(frame_w, frame_h, self.cfg.get("camera", {}))
        depth_result = self.depth_backbone.estimate(frame_bgr, modality=modality_name, camera_metadata=camera_metadata)
        depth_map = depth_result.depth_map_m

        estimates: list[RangeEstimate] = []
        for detection in detections:
            geometry = estimate_distance_from_bbox(
                detection.width,
                detection.height,
                camera_metadata,
                self.cfg.get("object_priors", {}),
            )
            depth_stats = extract_depth_patch_stats(depth_map, detection) if depth_map is not None else DepthPatchStats()
            estimate = self._combine_cues(
                detection=detection,
                geometry_nominal=geometry.nominal_m,
                geometry_min=geometry.min_m,
                geometry_max=geometry.max_m,
                depth_stats=depth_stats,
                used_fallback_camera=camera_metadata.used_fallback_intrinsics,
                depth_warning=depth_result.warning,
                frame_shape=(frame_h, frame_w),
            )
            estimates.append(estimate)
        return estimates

    def annotate_frame(self, frame_bgr: np.ndarray, estimates: list[RangeEstimate]) -> np.ndarray:
        if not self.cfg.get("hybrid_head", {}).get("overlay", True):
            return frame_bgr

        import cv2

        annotated = frame_bgr.copy()
        for estimate in estimates:
            if estimate.distance_m is None:
                continue
            text = f"{estimate.distance_m:.1f}m +/- {estimate.distance_std_m or 0.0:.1f}"
            if estimate.track_id is not None:
                text = f"ID {estimate.track_id} | {text}"
            x1 = int(max(0.0, estimate.x_center - (estimate.width / 2.0)))
            y1 = int(max(20.0, estimate.y_center - (estimate.height / 2.0) - 8.0))
            color = (0, 255, 0) if not estimate.low_confidence else (0, 200, 255)
            cv2.putText(
                annotated,
                text,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
        return annotated

    def _combine_cues(
        self,
        *,
        detection: DetectionInput,
        geometry_nominal: float | None,
        geometry_min: float | None,
        geometry_max: float | None,
        depth_stats: DepthPatchStats,
        used_fallback_camera: bool,
        depth_warning: str | None,
        frame_shape: tuple[int, int],
    ) -> RangeEstimate:
        hybrid_cfg = self.cfg.get("hybrid_head", {})
        frame_h, frame_w = frame_shape
        w_norm = detection.width / max(frame_w, 1)
        h_norm = detection.height / max(frame_h, 1)
        bbox_diag = math.hypot(detection.width, detection.height)
        bbox_quality = min(1.0, bbox_diag / max(float(hybrid_cfg.get("small_bbox_px", 18.0)), 1.0))

        depth_distance = depth_stats.median_m if depth_stats.valid_fraction >= float(hybrid_cfg.get("min_depth_valid_fraction", 0.15)) else None
        distance_m = geometry_nominal
        notes = []
        if geometry_nominal is not None:
            notes.append("geometry")
        if depth_distance is not None:
            notes.append("depth")

        if self.range_head is not None:
            features = build_feature_row(
                detection=detection,
                geometry_nominal=geometry_nominal,
                geometry_min=geometry_min,
                geometry_max=geometry_max,
                depth_stats=depth_stats,
                frame_shape=frame_shape,
                used_fallback_camera=used_fallback_camera,
            )
            feature_tensor = torch.tensor([features[name] for name in self.feature_columns], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, log_var, ordinal_logits = self.range_head(feature_tensor)
            distance_m = max(float(mean.squeeze().item()), 0.0)
            model_std = math.sqrt(math.exp(float(log_var.squeeze().item())))
            if depth_distance is None and geometry_nominal is None:
                geometry_min = max(distance_m - (2.0 * model_std), 0.0)
                geometry_max = distance_m + (2.0 * model_std)
            range_bin = range_bin_from_logits(ordinal_logits.squeeze(0), self.cfg.get("range_bins_m", [25.0, 75.0]))
            notes.append("learned_head")
        else:
            bbox_weight = float(hybrid_cfg.get("bbox_weight", 1.0))
            depth_weight = float(hybrid_cfg.get("depth_weight", 0.35)) if depth_distance is not None else 0.0
            total_weight = bbox_weight + depth_weight
            if total_weight > 0 and geometry_nominal is not None:
                distance_m = ((bbox_weight * geometry_nominal) + (depth_weight * (depth_distance or 0.0))) / total_weight
            elif depth_distance is not None:
                distance_m = depth_distance
            range_bin = classify_range_bin(distance_m, self.cfg.get("range_bins_m", [25.0, 75.0]))

        distance_std = None
        if distance_m is not None:
            spread_std = max(
                ((geometry_max or distance_m) - (geometry_min or distance_m)) / 4.0,
                float(hybrid_cfg.get("uncertainty_floor_m", 2.0)),
            )
            depth_std = depth_stats.std_m or 0.0
            distance_std = max(spread_std, depth_std)
            distance_std *= 1.0 + ((1.0 - min(max(detection.score, 0.0), 1.0)) * float(hybrid_cfg.get("low_score_penalty", 1.5)))
            if bbox_diag > 0:
                tiny_scale = min(
                    float(hybrid_cfg.get("tiny_bbox_penalty_cap", 3.0)),
                    max(1.0, float(hybrid_cfg.get("small_bbox_px", 18.0)) / bbox_diag),
                )
                distance_std *= tiny_scale
            if used_fallback_camera:
                distance_std *= float(hybrid_cfg.get("fallback_camera_penalty", 1.35))
        confidence = 0.0
        if distance_m is not None and distance_std is not None:
            confidence = min(
                0.99,
                detection.score * bbox_quality * (1.0 / (1.0 + (distance_std / max(distance_m, 1.0)))),
            )
            if used_fallback_camera:
                confidence *= 0.8
        low_confidence = confidence < float(hybrid_cfg.get("confidence_threshold", 0.35))

        if depth_warning:
            notes.append(depth_warning)
        if used_fallback_camera:
            notes.append("fallback_camera")

        return RangeEstimate(
            frame_index=detection.frame_index,
            class_id=detection.class_id,
            score=detection.score,
            x_center=detection.x_center,
            y_center=detection.y_center,
            width=detection.width,
            height=detection.height,
            track_id=detection.track_id,
            distance_m=distance_m,
            distance_std_m=distance_std,
            distance_confidence=confidence,
            range_bin=range_bin,
            low_confidence=low_confidence,
            geometric_distance_m=geometry_nominal,
            depth_distance_m=depth_distance,
            distance_min_m=geometry_min,
            distance_max_m=geometry_max,
            used_fallback_camera=used_fallback_camera,
            notes=";".join(notes) if notes else None,
        )

    def _load_range_head_if_available(self) -> None:
        head_cfg = self.cfg.get("range_head", {})
        if str(head_cfg.get("name", "heuristic")).lower() != "learned_mlp":
            return

        checkpoint_path = Path(str(head_cfg.get("checkpoint", "")))
        if not checkpoint_path.is_absolute():
            checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
        if not checkpoint_path.exists():
            if head_cfg.get("allow_missing", True):
                self.runtime_notes.append(f"Range-head checkpoint missing: {checkpoint_path}")
                return
            raise FileNotFoundError(f"Range-head checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.feature_columns = checkpoint.get("feature_columns", self.feature_columns)
        model = DroneRangeHead(
            input_dim=len(self.feature_columns),
            hidden_dim=int(checkpoint.get("hidden_dim", 128)),
            num_layers=int(checkpoint.get("num_layers", 3)),
            ordinal_bins=int(checkpoint.get("ordinal_bins", 3)),
            dropout=float(checkpoint.get("dropout", 0.1)),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self.range_head = model
        self.runtime_notes.append(f"Loaded learned range head: {checkpoint_path}")


DEFAULT_FEATURE_COLUMNS = [
    "score",
    "x_norm",
    "y_norm",
    "w_norm",
    "h_norm",
    "area_norm",
    "aspect_ratio",
    "bbox_diag_px",
    "geometric_distance_m",
    "distance_min_m",
    "distance_max_m",
    "depth_median_m",
    "depth_std_m",
    "depth_valid_fraction",
    "fallback_camera",
]


def extract_depth_patch_stats(depth_map_m: np.ndarray, detection: DetectionInput) -> DepthPatchStats:
    if depth_map_m is None or depth_map_m.size == 0:
        return DepthPatchStats()

    height, width = depth_map_m.shape[:2]
    x1 = int(max(0.0, detection.x_center - (detection.width / 2.0)))
    y1 = int(max(0.0, detection.y_center - (detection.height / 2.0)))
    x2 = int(min(width, detection.x_center + (detection.width / 2.0)))
    y2 = int(min(height, detection.y_center + (detection.height / 2.0)))
    if x2 <= x1 or y2 <= y1:
        return DepthPatchStats()

    patch = depth_map_m[y1:y2, x1:x2]
    if patch.size == 0:
        return DepthPatchStats()

    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return DepthPatchStats()

    return DepthPatchStats(
        median_m=float(np.median(valid)),
        mean_m=float(np.mean(valid)),
        std_m=float(np.std(valid)),
        valid_fraction=float(valid.size / patch.size),
    )


def build_feature_row(
    *,
    detection: DetectionInput,
    geometry_nominal: float | None,
    geometry_min: float | None,
    geometry_max: float | None,
    depth_stats: DepthPatchStats,
    frame_shape: tuple[int, int],
    used_fallback_camera: bool,
) -> dict[str, float]:
    frame_h, frame_w = frame_shape
    w_norm = detection.width / max(frame_w, 1)
    h_norm = detection.height / max(frame_h, 1)
    return {
        "score": float(detection.score),
        "x_norm": float(detection.x_center / max(frame_w, 1)),
        "y_norm": float(detection.y_center / max(frame_h, 1)),
        "w_norm": float(w_norm),
        "h_norm": float(h_norm),
        "area_norm": float(w_norm * h_norm),
        "aspect_ratio": float(detection.width / max(detection.height, 1e-6)),
        "bbox_diag_px": float(math.hypot(detection.width, detection.height)),
        "geometric_distance_m": float(geometry_nominal or 0.0),
        "distance_min_m": float(geometry_min or 0.0),
        "distance_max_m": float(geometry_max or 0.0),
        "depth_median_m": float(depth_stats.median_m or 0.0),
        "depth_std_m": float(depth_stats.std_m or 0.0),
        "depth_valid_fraction": float(depth_stats.valid_fraction),
        "fallback_camera": 1.0 if used_fallback_camera else 0.0,
    }


def range_bin_from_logits(logits: torch.Tensor, thresholds: list[float]) -> str:
    if logits.numel() == 0:
        return "unknown"
    label_index = int(torch.argmax(logits).item())
    labels = ["close", "medium", "distant"]
    if label_index < len(labels):
        return labels[label_index]
    return classify_range_bin(None, thresholds)
