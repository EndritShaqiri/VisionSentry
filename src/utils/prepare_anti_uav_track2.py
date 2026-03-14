from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SequenceSummary:
    name: str
    frame_count: int
    labeled_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Anti-UAV Track 2 raw training data into YOLO train/val folders."
    )
    parser.add_argument(
        "--raw-train-dir",
        type=str,
        default="data/raw/train",
        help="Path to extracted Anti-UAV train directory.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/thermal_uav",
        help="YOLO dataset root directory.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of labeled sequences reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sequence split.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of using hard links when possible.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear existing YOLO train/val image and label folders before conversion.",
    )
    return parser.parse_args()


def reset_split_dirs(output_root: Path, clear_output: bool) -> None:
    for split in ("train", "val"):
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        if clear_output:
            shutil.rmtree(image_dir, ignore_errors=True)
            shutil.rmtree(label_dir, ignore_errors=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)


def list_sequence_dirs(raw_train_dir: Path) -> list[Path]:
    return sorted([p for p in raw_train_dir.iterdir() if p.is_dir()])


def list_frame_files(sequence_dir: Path) -> list[Path]:
    return sorted([p for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def safe_link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def load_annotations(label_path: Path) -> dict:
    with label_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_sequences(sequence_dirs: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    labeled_sequences = [seq for seq in sequence_dirs if (seq / "IR_label.json").exists()]
    rng = random.Random(seed)
    shuffled = labeled_sequences[:]
    rng.shuffle(shuffled)

    if not shuffled:
        return [], []

    val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 and val_ratio > 0 else 0
    val_sequences = sorted(shuffled[:val_count])
    train_sequences = sorted(shuffled[val_count:])

    if not train_sequences and val_sequences:
        train_sequences.append(val_sequences.pop())

    return train_sequences, val_sequences


def build_yolo_label(rect: list[float], image_w: int, image_h: int) -> str | None:
    if len(rect) != 4:
        return None

    x, y, w, h = rect
    if w <= 0 or h <= 0 or image_w <= 0 or image_h <= 0:
        return None

    x_center = (x + (w / 2.0)) / image_w
    y_center = (y + (h / 2.0)) / image_h
    width = w / image_w
    height = h / image_h

    values = [x_center, y_center, width, height]
    if any(v < 0.0 or v > 1.0 for v in values):
        return None

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def convert_sequence(
    sequence_dir: Path,
    split: str,
    output_root: Path,
    copy_images: bool,
) -> SequenceSummary:
    label_path = sequence_dir / "IR_label.json"
    annotations = load_annotations(label_path)
    exists = annotations.get("exist", [])
    gt_rect = annotations.get("gt_rect", [])
    frame_files = list_frame_files(sequence_dir)

    if len(exists) != len(gt_rect):
        raise ValueError(f"Annotation length mismatch in {sequence_dir}")

    if not frame_files:
        return SequenceSummary(name=sequence_dir.name, frame_count=0, labeled_count=0)

    first_image = cv2.imread(str(frame_files[0]))
    if first_image is None:
        raise ValueError(f"Failed to read first frame in {sequence_dir}")
    image_h, image_w = first_image.shape[:2]

    labeled_count = 0
    for frame_idx, frame_path in enumerate(frame_files):
        output_name = f"{sequence_dir.name}__{frame_path.stem}{frame_path.suffix.lower()}"
        output_image_path = output_root / "images" / split / output_name
        output_label_path = output_root / "labels" / split / f"{sequence_dir.name}__{frame_path.stem}.txt"

        safe_link_or_copy(frame_path, output_image_path, copy_images=copy_images)

        label_text = ""
        if frame_idx < len(exists) and exists[frame_idx] == 1:
            yolo_line = build_yolo_label(gt_rect[frame_idx], image_w=image_w, image_h=image_h)
            if yolo_line:
                label_text = yolo_line
                labeled_count += 1
        output_label_path.write_text(label_text, encoding="utf-8")

    return SequenceSummary(name=sequence_dir.name, frame_count=len(frame_files), labeled_count=labeled_count)


def main() -> None:
    args = parse_args()
    raw_train_dir = Path(args.raw_train_dir)
    output_root = Path(args.output_root)

    if not raw_train_dir.exists():
        raise FileNotFoundError(f"Raw Anti-UAV train directory not found: {raw_train_dir}")

    reset_split_dirs(output_root=output_root, clear_output=args.clear_output)
    sequence_dirs = list_sequence_dirs(raw_train_dir)
    train_sequences, val_sequences = split_sequences(
        sequence_dirs=sequence_dirs,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Raw train dir: {raw_train_dir.resolve()}")
    print(f"YOLO output root: {output_root.resolve()}")
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")

    train_summaries = [convert_sequence(seq, "train", output_root, args.copy_images) for seq in train_sequences]
    val_summaries = [convert_sequence(seq, "val", output_root, args.copy_images) for seq in val_sequences]

    train_frames = sum(item.frame_count for item in train_summaries)
    val_frames = sum(item.frame_count for item in val_summaries)
    train_labels = sum(item.labeled_count for item in train_summaries)
    val_labels = sum(item.labeled_count for item in val_summaries)

    print("\nConversion summary:")
    print(f"  train frames: {train_frames}")
    print(f"  train labeled frames: {train_labels}")
    print(f"  val frames: {val_frames}")
    print(f"  val labeled frames: {val_labels}")
    print("\n[OK] Anti-UAV Track 2 train set converted to YOLO format.")
    print(f"[OK] Images: {(output_root / 'images').resolve()}")
    print(f"[OK] Labels: {(output_root / 'labels').resolve()}")


if __name__ == "__main__":
    main()
