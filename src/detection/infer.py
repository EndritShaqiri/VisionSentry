from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import yaml
from ultralytics import YOLO

from distance_estimation import DetectionInput, RangeEstimator
from src.utils.paths import get_video_fps, make_run_dir, resolve_existing_path

DEFAULTS: dict[str, Any] = {
    "weights": "runs/detect/yolo12n_thermal_uav/weights/best.pt",
    "source": "data/sample.mp4",
    "imgsz": 960,
    "conf": 0.25,
    "iou": 0.50,
    "device": "0",
    "project": "runs/predict",
    "name": "thermal_detect",
    "exist_ok": False,
    "save_video": True,
    "save_frames": False,
    "save_txt": False,
    "fps": 30.0,
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
    parser = argparse.ArgumentParser(description="Run detector-only inference for single or multi-UAV thermal data.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--weights", type=str, default=None, help="Path to detector weights.")
    parser.add_argument("--source", type=str, default=None, help="Source video file or frame folder.")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold.")
    parser.add_argument("--device", type=str, default=None, help='Device string: "0", "0,1", "cpu".')
    parser.add_argument("--project", type=str, default=None, help="Output project directory.")
    parser.add_argument("--name", type=str, default=None, help="Run name.")
    parser.add_argument("--exist_ok", type=str2bool, default=None, help="Allow overwriting run folder.")
    parser.add_argument("--save_video", type=str2bool, default=None, help="Save annotated video.")
    parser.add_argument("--save_frames", type=str2bool, default=None, help="Save annotated frames.")
    parser.add_argument("--save_txt", type=str2bool, default=None, help="Save per-frame detections CSV.")
    parser.add_argument("--fps", type=float, default=None, help="Fallback FPS for frame-folder inputs.")
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


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    weights = resolve_existing_path(cfg["weights"], description="weights")
    source = resolve_existing_path(cfg["source"], description="source")
    save_dir = make_run_dir(cfg["project"], cfg["name"], exist_ok=cfg["exist_ok"])

    frames_dir = save_dir / "frames"
    if cfg["save_frames"]:
        frames_dir.mkdir(parents=True, exist_ok=True)

    detections_csv = save_dir / "detections.csv"
    det_writer = None
    ranging_estimator = None
    ranging_runtime_path = None
    if cfg["ranging"]:
        ranging_estimator = RangeEstimator.from_paths(
            config_path=cfg["ranging_config"],
            camera_config_path=cfg.get("camera_config"),
            modality=cfg.get("ranging_modality"),
        )
        ranging_runtime_path = ranging_estimator.save_runtime_config(save_dir)

    if cfg["save_txt"] or cfg["ranging"]:
        det_writer = detections_csv.open("w", encoding="utf-8")
        if ranging_estimator is not None:
            det_writer.write(
                "frame,track_id,class_id,x,y,w,h,score,distance_m,distance_std_m,distance_confidence,"
                "range_bin,low_confidence,distance_min_m,distance_max_m,geometric_distance_m,"
                "depth_distance_m,used_fallback_camera,notes\n"
            )
        else:
            det_writer.write("frame,class_id,x,y,w,h,score\n")

    print("Inference config:")
    for key in sorted(cfg.keys()):
        print(f"  {key}: {cfg[key]}")
    print(f"  save_dir: {save_dir}")

    model = YOLO(str(weights))
    stream = model.predict(
        source=str(source),
        imgsz=cfg["imgsz"],
        conf=cfg["conf"],
        iou=cfg["iou"],
        device=cfg["device"],
        stream=True,
        verbose=False,
    )

    video_writer = None
    output_video_path = save_dir / "detected.mp4"
    fps = get_video_fps(source, fallback_fps=cfg["fps"])

    for frame_idx, result in enumerate(stream, start=1):
        estimates = []
        if result.boxes is not None and len(result.boxes) > 0:
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
                )
                for box, score, cls_id in zip(xywh, confs, class_ids)
            ]
            if ranging_estimator is not None:
                estimates = ranging_estimator.estimate_detections(
                    result.orig_img,
                    detections,
                    modality=cfg["ranging_modality"],
                )

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

        if det_writer is not None and result.boxes is not None and len(result.boxes) > 0:
            if ranging_estimator is not None and estimates:
                for estimate in estimates:
                    row = estimate.as_csv_row()
                    det_writer.write(
                        f"{row['frame']},{row['track_id']},{row['class_id']},{row['x']},{row['y']},{row['w']},{row['h']},"
                        f"{row['score']},{row['distance_m']},{row['distance_std_m']},{row['distance_confidence']},"
                        f"{row['range_bin']},{row['low_confidence']},{row['distance_min_m']},{row['distance_max_m']},"
                        f"{row['geometric_distance_m']},{row['depth_distance_m']},{row['used_fallback_camera']},{row['notes']}\n"
                    )
            else:
                xywh = result.boxes.xywh.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.int().cpu().numpy()
                for box, score, cls_id in zip(xywh, confs, class_ids):
                    x_c, y_c, bw, bh = box.tolist()
                    det_writer.write(
                        f"{frame_idx},{int(cls_id)},{x_c:.4f},{y_c:.4f},{bw:.4f},{bh:.4f},{float(score):.6f}\n"
                    )

    if video_writer is not None:
        video_writer.release()
    if det_writer is not None:
        det_writer.close()

    if cfg["save_video"]:
        print(f"[OK] Annotated video: {output_video_path.resolve()}")
    if cfg["save_frames"]:
        print(f"[OK] Annotated frames dir: {frames_dir.resolve()}")
    if cfg["save_txt"] or cfg["ranging"]:
        print(f"[OK] Detections CSV: {detections_csv.resolve()}")
    if ranging_runtime_path is not None:
        print(f"[OK] Ranging runtime config: {ranging_runtime_path.resolve()}")


if __name__ == "__main__":
    main()
