from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import yaml
from ultralytics import YOLO

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
}


def str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector-only inference for thermal UAV data.")
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
    if cfg["save_txt"]:
        det_writer = detections_csv.open("w", encoding="utf-8")
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
        annotated = result.plot()
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
    if cfg["save_txt"]:
        print(f"[OK] Detections CSV: {detections_csv.resolve()}")


if __name__ == "__main__":
    main()
