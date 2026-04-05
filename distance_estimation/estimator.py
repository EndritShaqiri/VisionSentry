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
            quality = float(estimate.quality_score or 0.0)
            if self.should_use_range_bin_overlay(quality) and estimate.range_bin != "unknown":
                text = estimate.range_bin.replace("_", " ").title()
            else:
                display_std = estimate.display_distance_std_m if estimate.display_distance_std_m is not None else estimate.distance_std_m
                text = f"{estimate.distance_m:.1f}m"
                if display_std is not None:
                    text = f"{text} +/- {display_std:.1f}"
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

    def build_display_std(self, *, distance_m: float | None, raw_std_m: float | None, quality_score: float | None) -> float | None:
        if distance_m is None or raw_std_m is None:
            return raw_std_m

        policy_cfg = self.cfg.get("uncertainty_policy", {})
        if not policy_cfg.get("use_quality_aware_display", True):
            return raw_std_m

        quality = min(1.0, max(0.0, float(quality_score or 0.0)))
        precise_floor = float(policy_cfg.get("min_quality_for_precise_overlay", 0.80))
        meter_floor = float(policy_cfg.get("min_quality_for_meter_overlay", 0.50))
        if quality >= precise_floor:
            target = float(policy_cfg.get("target_relative_std_high_quality", 0.15)) * distance_m
            return min(raw_std_m, target)
        if quality >= meter_floor:
            target = float(policy_cfg.get("target_relative_std_medium_quality", 0.25)) * distance_m
            return min(raw_std_m, target)
        return raw_std_m

    def should_use_range_bin_overlay(self, quality_score: float | None) -> bool:
        policy_cfg = self.cfg.get("uncertainty_policy", {})
        if not policy_cfg.get("use_range_bins_when_low_quality", True):
            return False
        quality = min(1.0, max(0.0, float(quality_score or 0.0)))
        return quality < float(policy_cfg.get("min_quality_for_meter_overlay", 0.50))

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
            range_bin = range_bin_from_logits(ordinal_logits.squeeze(0), self.cfg.get("range_bins_m", [50.0, 150.0]))
            notes.append("learned_head")
        else:
            bbox_weight = float(hybrid_cfg.get("bbox_weight", 1.0))
            depth_weight = float(hybrid_cfg.get("depth_weight", 0.35)) if depth_distance is not None else 0.0
            total_weight = bbox_weight + depth_weight
            if total_weight > 0 and geometry_nominal is not None:
                distance_m = ((bbox_weight * geometry_nominal) + (depth_weight * (depth_distance or 0.0))) / total_weight
            elif depth_distance is not None:
                distance_m = depth_distance
            range_bin = classify_range_bin(distance_m, self.cfg.get("range_bins_m", [50.0, 150.0]))

        raw_distance_std = None
        if distance_m is not None:
            spread_std = max(
                ((geometry_max or distance_m) - (geometry_min or distance_m)) / 4.0,
                float(hybrid_cfg.get("uncertainty_floor_m", 2.0)),
            )
            depth_std = depth_stats.std_m or 0.0
            raw_distance_std = max(spread_std, depth_std)
            low_score_penalty = float(hybrid_cfg.get("low_score_penalty", 1.1))
            raw_distance_std *= 1.0 + (
                (1.0 - min(max(detection.score, 0.0), 1.0)) * max(0.0, low_score_penalty - 1.0)
            )
            if bbox_diag > 0:
                tiny_scale = min(
                    float(hybrid_cfg.get("tiny_bbox_penalty_cap", 3.0)),
                    max(1.0, float(hybrid_cfg.get("small_bbox_px", 18.0)) / bbox_diag),
                )
                raw_distance_std *= tiny_scale
            if used_fallback_camera:
                raw_distance_std *= float(hybrid_cfg.get("fallback_camera_penalty", 1.2))
            max_relative_std_raw = float(self.cfg.get("uncertainty_policy", {}).get("max_relative_std_raw", 0.35))
            raw_distance_std = min(raw_distance_std, max_relative_std_raw * max(distance_m, 1.0))

        quality_score = self._compute_quality_score(
            detection_score=detection.score,
            bbox_diag=bbox_diag,
            bbox_quality=bbox_quality,
            used_fallback_camera=used_fallback_camera,
        )
        display_distance_std = self.build_display_std(
            distance_m=distance_m,
            raw_std_m=raw_distance_std,
            quality_score=quality_score,
        )

        confidence = 0.0
        if distance_m is not None and raw_distance_std is not None:
            confidence = min(
                0.99,
                quality_score * (1.0 / (1.0 + (raw_distance_std / max(distance_m, 1.0)))),
            )
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
            distance_std_m=raw_distance_std,
            display_distance_std_m=display_distance_std,
            distance_confidence=confidence,
            quality_score=quality_score,
            range_bin=range_bin,
            low_confidence=low_confidence,
            geometric_distance_m=geometry_nominal,
            depth_distance_m=depth_distance,
            distance_min_m=geometry_min,
            distance_max_m=geometry_max,
            used_fallback_camera=used_fallback_camera,
            notes=";".join(notes) if notes else None,
        )

    def _compute_quality_score(
        self,
        *,
        detection_score: float,
        bbox_diag: float,
        bbox_quality: float,
        used_fallback_camera: bool,
    ) -> float:
        small_bbox_px = max(float(self.cfg.get("hybrid_head", {}).get("small_bbox_px", 20.0)), 1.0)
        score_quality = min(1.0, max(0.0, float(detection_score)))
        bbox_quality = min(1.0, max(0.0, float(bbox_quality)))
        quality = (0.5 * score_quality) + (0.5 * bbox_quality)
        if used_fallback_camera:
            quality *= 0.9
        if bbox_diag < small_bbox_px:
            quality *= max(0.75, bbox_quality)
        return min(1.0, max(0.0, quality))

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
