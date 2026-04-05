from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import yaml
from ultralytics import YOLO

from distance_estimation import DetectionInput, RangeEstimator, smooth_track_ranges
from src.utils.paths import get_video_fps, make_run_dir, resolve_existing_path

DEFAULTS: dict[str, Any] = {
    "weights": "runs/detect/yolo12n_thermal_uav/weights/best.pt",
    "source": "data/sample.mp4",
    "tracker": "configs/tracker_botsort.yaml",
    "conf": 0.25,
    "iou": 0.50,
    "imgsz": 960,
    "device": "0",
    "project": "runs/track",
    "name": "thermal_track",
    "exist_ok": False,
    "save_video": True,
    "save_frames": True,
    "mot_txt_name": "tracks_mot.txt",
    "fps": 30.0,
    "with_reid": None,
    "class_id": 0,
    "ranging": False,
    "ranging_config": "distance_estimation/configs/ranging.yaml",
    "camera_config": None,
    "ranging_modality": "ir",
}


def str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BoT-SORT UAV tracking for single or multi-UAV sequences.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--weights", type=str, default=None, help="Path to detector weights.")
    parser.add_argument("--source", type=str, default=None, help="Source video file or frame folder.")
    parser.add_argument("--tracker", type=str, default=None, help="BoT-SORT tracker YAML path.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size.")
    parser.add_argument("--device", type=str, default=None, help='Device string: "0", "0,1", "cpu".')
    parser.add_argument("--project", type=str, default=None, help="Output project directory.")
    parser.add_argument("--name", type=str, default=None, help="Run name.")
    parser.add_argument("--exist_ok", type=str2bool, default=None, help="Allow overwriting run folder.")
    parser.add_argument("--save_video", type=str2bool, default=None, help="Save annotated tracking video.")
    parser.add_argument("--save_frames", type=str2bool, default=None, help="Save annotated frames.")
    parser.add_argument("--mot_txt_name", type=str, default=None, help="MOT output filename.")
    parser.add_argument("--fps", type=float, default=None, help="Fallback FPS for frame-folder inputs.")
    parser.add_argument("--with_reid", type=str2bool, default=None, help="Override tracker with_reid setting.")
    parser.add_argument("--class_id", type=int, default=None, help="Class filter. Use 0 for single-class UAV.")
    parser.add_argument("--ranging", type=str2bool, default=None, help="Enable distance estimation.")
    parser.add_argument("--ranging_config", type=str, default=None, help="Distance-estimation YAML config path.")
    parser.add_argument("--camera_config", type=str, default=None, help="Optional camera profile YAML path.")
    parser.add_argument("--ranging_modality", type=str, default=None, choices=["ir", "rgb"], help="Distance-estimation modality.")
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


def write_runtime_tracker_config(
    tracker_cfg_path: Path,
    save_dir: Path,
    with_reid_override: bool | None = None,
) -> tuple[Path, dict[str, Any]]:
    tracker_cfg = load_yaml(tracker_cfg_path)
    if with_reid_override is not None:
        tracker_cfg["with_reid"] = with_reid_override

    runtime_cfg_path = save_dir / "tracker_runtime.yaml"
    with runtime_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(tracker_cfg, f, sort_keys=False)
    return runtime_cfg_path, tracker_cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    weights = resolve_existing_path(cfg["weights"], description="weights")
    source = resolve_existing_path(cfg["source"], description="source")
    tracker_cfg_path = resolve_existing_path(cfg["tracker"], description="tracker config")

    save_dir = make_run_dir(cfg["project"], cfg["name"], exist_ok=cfg["exist_ok"])
    frames_dir = save_dir / "frames"
    if cfg["save_frames"]:
        frames_dir.mkdir(parents=True, exist_ok=True)

    runtime_tracker_cfg, tracker_cfg = write_runtime_tracker_config(
        tracker_cfg_path=tracker_cfg_path,
        save_dir=save_dir,
        with_reid_override=cfg["with_reid"],
    )

    mot_path = save_dir / cfg["mot_txt_name"]
    mot_file = mot_path.open("w", encoding="utf-8")
    track_ranges_path = save_dir / "track_ranges.csv"
    ranging_estimator = None
    ranging_runtime_path = None
    all_track_estimates = []
    if cfg["ranging"]:
        ranging_estimator = RangeEstimator.from_paths(
            config_path=cfg["ranging_config"],
            camera_config_path=cfg.get("camera_config"),
            modality=cfg.get("ranging_modality"),
        )
        ranging_runtime_path = ranging_estimator.save_runtime_config(save_dir)

    print("Tracking config:")
    for key in sorted(cfg.keys()):
        print(f"  {key}: {cfg[key]}")
    print(f"  runtime_tracker_config: {runtime_tracker_cfg}")
    print(f"  with_reid_effective: {tracker_cfg.get('with_reid', False)}")
    print(f"  save_dir: {save_dir}")

    model = YOLO(str(weights))
    classes = [cfg["class_id"]] if cfg["class_id"] is not None else None
    stream = model.track(
        source=str(source),
        tracker=str(runtime_tracker_cfg),
        conf=cfg["conf"],
        iou=cfg["iou"],
        imgsz=cfg["imgsz"],
        device=cfg["device"],
        classes=classes,
        persist=True,
        stream=True,
        verbose=False,
    )

    fps = get_video_fps(source, fallback_fps=cfg["fps"])
    video_writer = None
    output_video_path = save_dir / "tracked.mp4"

    for frame_idx, result in enumerate(stream, start=1):
        estimates = []
        if result.boxes is not None and result.boxes.id is not None and len(result.boxes) > 0:
            ids = result.boxes.id.int().cpu().numpy()
            xywh = result.boxes.xywh.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().numpy()
            detections = [
                DetectionInput(
                    frame_index=frame_idx,
                    class_id=int(cls_id),
                    score=float(score),
                    x_center=float(box[0]),
                    y_center=float(box[1]),
                    width=float(box[2]),
                    height=float(box[3]),
                    track_id=int(track_id),
                )
                for track_id, box, score, cls_id in zip(ids, xywh, confs, class_ids)
            ]
            if ranging_estimator is not None:
                estimates = ranging_estimator.estimate_detections(
                    result.orig_img,
                    detections,
                    modality=cfg["ranging_modality"],
                )
                all_track_estimates.extend(estimates)

        annotated = result.plot(line_width=1, font_size=10)
        if ranging_estimator is not None and estimates:
            annotated = ranging_estimator.annotate_frame(annotated, estimates)
        h, w = annotated.shape[:2]

        if cfg["save_video"] and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        if video_writer is not None:
            video_writer.write(annotated)

        if cfg["save_frames"]:
            frame_path = frames_dir / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), annotated)

        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            continue

        ids = boxes.id.int().cpu().numpy()
        xywh = boxes.xywh.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for track_id, box, score in zip(ids, xywh, confs):
            x_c, y_c, bw, bh = box.tolist()
            x_tl = x_c - (bw / 2.0)
            y_tl = y_c - (bh / 2.0)
            mot_file.write(
                f"{frame_idx},{int(track_id)},{x_tl:.2f},{y_tl:.2f},{bw:.2f},{bh:.2f},{float(score):.6f},-1,-1,-1\n"
            )

    if video_writer is not None:
        video_writer.release()
    mot_file.close()

    if ranging_estimator is not None and all_track_estimates:
        temporal_cfg = ranging_estimator.cfg.get("temporal", {})
        smoothed_tracks = smooth_track_ranges(
            all_track_estimates,
            process_noise_m2=float(temporal_cfg.get("process_noise_m2", 9.0)),
            min_measurement_std_m=float(temporal_cfg.get("min_measurement_std_m", 2.0)),
            measurement_noise_scale=float(temporal_cfg.get("measurement_noise_scale", 1.0)),
            range_bins_m=ranging_estimator.cfg.get("range_bins_m", [25.0, 75.0]),
        )
        with track_ranges_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "track_id",
                    "distance_m_raw",
                    "distance_m_filtered",
                    "distance_m_refined",
                    "distance_std_m",
                    "distance_confidence",
                    "range_bin",
                    "low_confidence",
                ],
            )
            writer.writeheader()
            for track_id in sorted(smoothed_tracks.keys()):
                for item in smoothed_tracks[track_id]:
                    writer.writerow(item.as_csv_row())

    if cfg["save_video"]:
        print(f"[OK] Tracked video: {output_video_path.resolve()}")
    if cfg["save_frames"]:
        print(f"[OK] Annotated frames dir: {frames_dir.resolve()}")
    print(f"[OK] MOT results: {mot_path.resolve()}")
    if ranging_estimator is not None and all_track_estimates:
        print(f"[OK] Track ranges CSV: {track_ranges_path.resolve()}")
    if ranging_runtime_path is not None:
        print(f"[OK] Ranging runtime config: {ranging_runtime_path.resolve()}")


if __name__ == "__main__":
    main()
