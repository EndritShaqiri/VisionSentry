from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypeVar

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
MODALITY_ALIASES = {
    "ir": ("ir", "infrared", "thermal"),
    "rgb": ("rgb", "visible", "vis", "color", "colour"),
}
MODALITY_LABEL_FILENAMES = {
    "ir": ("IR_label.json", "infrared_label.json", "thermal_label.json"),
    "rgb": ("RGB_label.json", "visible_label.json", "vis_label.json", "color_label.json", "colour_label.json"),
}
T = TypeVar("T")


@dataclass
class SequenceSummary:
    name: str
    frame_count: int
    labeled_count: int
    object_count: int
    annotation_mode: str


@dataclass(frozen=True)
class SequenceSource:
    name: str
    media_path: Path
    annotation_path: Path | None
    media_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Anti-UAV raw data into YOLO train/val/test folders for single or multi-UAV tasks."
    )
    parser.add_argument(
        "--raw-train-dir",
        type=str,
        default="data/raw/train",
        help="Path to extracted Anti-UAV train directory.",
    )
    parser.add_argument(
        "--raw-test-dir",
        type=str,
        default=None,
        help="Optional extracted Anti-UAV test directory. Frames are copied/linked into images/test without labels.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/thermal_uav",
        help="YOLO dataset root directory.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="ir",
        choices=sorted(MODALITY_ALIASES.keys()),
        help="Input modality for frame-based Anti-UAV sequence layouts.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["auto", "single", "multi"],
        help="Expected task style. 'auto' detects annotation structure per sequence.",
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
        "--split-manifest",
        type=str,
        default=None,
        help="Optional JSON split manifest path. Existing manifests are reused; otherwise a new one is written.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear existing YOLO train/val/test image and label folders before conversion.",
    )
    return parser.parse_args()


def reset_split_dirs(output_root: Path, clear_output: bool) -> None:
    for split in ("train", "val", "test"):
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        if clear_output:
            shutil.rmtree(image_dir, ignore_errors=True)
            shutil.rmtree(label_dir, ignore_errors=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)


def list_sequence_dirs(raw_dir: Path) -> list[Path]:
    return sorted([p for p in raw_dir.iterdir() if p.is_dir()])


def list_frame_files(sequence_dir: Path) -> list[Path]:
    return sorted([p for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def list_video_files(video_dir: Path) -> list[Path]:
    return sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])


def safe_link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def sort_key(item: T) -> str:
    return getattr(item, "name", str(item))


def contains_modality_token(value: str, modality: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in MODALITY_ALIASES[modality])


def default_split_manifest_path(raw_train_dir: Path, seed: int) -> Path:
    return raw_train_dir.parent / f"{raw_train_dir.name}_split_seed{seed}.json"


def load_split_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_split_manifest(
    manifest_path: Path,
    raw_train_dir: Path,
    modality: str,
    val_ratio: float,
    seed: int,
    train_sequences: list[T],
    val_sequences: list[T],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw_train_dir": str(raw_train_dir.resolve()),
        "modality": modality,
        "seed": seed,
        "val_ratio": val_ratio,
        "train": [item.name for item in train_sequences],
        "val": [item.name for item in val_sequences],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def split_sequences_with_manifest(
    sequence_items: list[T],
    *,
    val_ratio: float,
    seed: int,
    manifest_path: Path,
    raw_train_dir: Path,
    modality: str,
) -> tuple[list[T], list[T], str]:
    items_by_name = {item.name: item for item in sequence_items}

    if manifest_path.exists():
        manifest = load_split_manifest(manifest_path)
        train_names = manifest.get("train", [])
        val_names = manifest.get("val", [])
        expected_names = set(items_by_name)
        manifest_names = set(train_names) | set(val_names)
        missing = sorted(expected_names - manifest_names)
        extra = sorted(manifest_names - expected_names)
        if missing or extra:
            raise ValueError(
                f"Split manifest {manifest_path} does not match the discovered sequences. "
                f"missing={missing}, extra={extra}"
            )
        train_sequences = [items_by_name[name] for name in train_names]
        val_sequences = [items_by_name[name] for name in val_names]
        return train_sequences, val_sequences, "reused"

    train_sequences, val_sequences = split_sequences(sequence_items, val_ratio=val_ratio, seed=seed)
    write_split_manifest(
        manifest_path=manifest_path,
        raw_train_dir=raw_train_dir,
        modality=modality,
        val_ratio=val_ratio,
        seed=seed,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
    )
    return train_sequences, val_sequences, "created"


def split_sequences(sequence_items: list[T], val_ratio: float, seed: int) -> tuple[list[T], list[T]]:
    rng = random.Random(seed)
    shuffled = sequence_items[:]
    rng.shuffle(shuffled)

    if not shuffled:
        return [], []

    val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 and val_ratio > 0 else 0
    val_sequences = sorted(shuffled[:val_count], key=sort_key)
    train_sequences = sorted(shuffled[val_count:], key=sort_key)

    if not train_sequences and val_sequences:
        train_sequences.append(val_sequences.pop())

    return train_sequences, val_sequences


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_yolo_line(rect: Iterable[float], image_w: int, image_h: int) -> str | None:
    values = list(rect)
    if len(values) != 4 or image_w <= 0 or image_h <= 0:
        return None

    x, y, w, h = map(float, values)
    if w <= 0 or h <= 0:
        return None

    x1 = max(0.0, min(x, float(image_w)))
    y1 = max(0.0, min(y, float(image_h)))
    x2 = max(0.0, min(x + w, float(image_w)))
    y2 = max(0.0, min(y + h, float(image_h)))
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0.0 or clipped_h <= 0.0:
        return None

    x_center = (x1 + (clipped_w / 2.0)) / image_w
    y_center = (y1 + (clipped_h / 2.0)) / image_h
    width = clipped_w / image_w
    height = clipped_h / image_h
    normalized = [x_center, y_center, width, height]
    if any(v < 0.0 or v > 1.0 for v in normalized):
        return None
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def is_rect_candidate(value: object) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value)


def parse_single_or_multi_json(annotation_path: Path, task: str) -> tuple[list[list[list[float]]], str]:
    payload = load_json(annotation_path)
    gt_rect = payload.get("gt_rect")
    if not isinstance(gt_rect, list):
        raise ValueError(f"Unsupported {annotation_path.name} schema in {annotation_path.parent}")

    # Track 2 style: one bbox per frame with optional exist flags.
    if gt_rect and is_rect_candidate(gt_rect[0]):
        exists = payload.get("exist", [1] * len(gt_rect))
        frames: list[list[list[float]]] = []
        for idx, rect in enumerate(gt_rect):
            visible = exists[idx] == 1 if idx < len(exists) else True
            frames.append([rect] if visible else [])
        return frames, f"{annotation_path.stem}_single_json"

    # Generic multi-target JSON: one list of bboxes per frame.
    if gt_rect and isinstance(gt_rect[0], list):
        frames_multi: list[list[list[float]]] = []
        for frame_entry in gt_rect:
            if is_rect_candidate(frame_entry):
                frames_multi.append([frame_entry])
                continue
            if isinstance(frame_entry, list):
                boxes = [box for box in frame_entry if is_rect_candidate(box)]
                frames_multi.append(boxes)
                continue
            frames_multi.append([])

        if task == "single":
            frames_multi = [[frame_boxes[0]] if frame_boxes else [] for frame_boxes in frames_multi]
        return frames_multi, f"{annotation_path.stem}_multi_json"

    return [], f"{annotation_path.stem}_empty_json"


def parse_mot_annotation_file(annotation_path: Path, task: str) -> tuple[list[list[list[float]]], str]:
    frames: dict[int, list[list[float]]] = {}
    with annotation_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 6:
                continue
            try:
                frame_idx = int(float(row[0]))
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])
                mark = float(row[6]) if len(row) > 6 else 1.0
            except ValueError:
                continue
            if frame_idx < 1 or mark <= 0:
                continue
            frames.setdefault(frame_idx, []).append([x, y, w, h])

    if not frames:
        return [], "mot_txt"

    max_frame_idx = max(frames.keys())
    ordered_frames = [frames.get(frame_idx, []) for frame_idx in range(1, max_frame_idx + 1)]
    if task == "single":
        ordered_frames = [[frame_boxes[0]] if frame_boxes else [] for frame_boxes in ordered_frames]
    return ordered_frames, "mot_txt"


def parse_mot_annotations(sequence_dir: Path, task: str) -> tuple[list[list[list[float]]], str] | None:
    candidates = [
        sequence_dir / "gt" / "gt.txt",
        sequence_dir / "gt.txt",
        sequence_dir / "annotations.txt",
    ]
    annotation_path = next((path for path in candidates if path.exists()), None)
    if annotation_path is None:
        return None

    return parse_mot_annotation_file(annotation_path, task=task)


def resolve_sequence_annotation_path(sequence_dir: Path, modality: str) -> Path | None:
    for candidate_name in MODALITY_LABEL_FILENAMES[modality]:
        candidate = sequence_dir / candidate_name
        if candidate.exists():
            return candidate

    json_candidates = sorted(path for path in sequence_dir.glob("*.json") if path.is_file())
    token_matches = [path for path in json_candidates if contains_modality_token(path.stem, modality)]
    if len(token_matches) == 1:
        return token_matches[0]

    label_jsons = [path for path in json_candidates if path.name.lower().endswith("_label.json")]
    if len(label_jsons) == 1:
        return label_jsons[0]

    return None


def resolve_sequence_media_source(sequence_dir: Path, modality: str) -> tuple[Path, str]:
    preferred_frame_dirs = [
        path
        for path in sorted(sequence_dir.iterdir())
        if path.is_dir() and contains_modality_token(path.name, modality) and list_frame_files(path)
    ]
    if preferred_frame_dirs:
        return preferred_frame_dirs[0], "frame_dir"

    top_level_frames = list_frame_files(sequence_dir)
    if top_level_frames:
        return sequence_dir, "frame_dir"

    preferred_videos = [
        path
        for path in list_video_files(sequence_dir)
        if contains_modality_token(path.stem, modality) or contains_modality_token(path.name, modality)
    ]
    if preferred_videos:
        return preferred_videos[0], "video_file"

    all_videos = list_video_files(sequence_dir)
    if len(all_videos) == 1:
        return all_videos[0], "video_file"

    raise FileNotFoundError(f"Could not resolve {modality} media for sequence: {sequence_dir}")


def load_annotations_for_source(source: SequenceSource, task: str) -> tuple[list[list[list[float]]], str]:
    if source.annotation_path is None:
        raise FileNotFoundError(f"No annotation path provided for sequence source: {source.name}")

    if source.annotation_path.suffix.lower() == ".json":
        return parse_single_or_multi_json(source.annotation_path, task=task)
    return parse_mot_annotation_file(source.annotation_path, task=task)


def resolve_track3_video_layout(raw_dir: Path) -> tuple[Path | None, Path | None]:
    if (raw_dir / "TrainVideos").is_dir():
        label_dir = raw_dir / "TrainLabels"
        return raw_dir / "TrainVideos", label_dir if label_dir.exists() else None

    if (raw_dir / "TestVideos").is_dir():
        label_dir = raw_dir / "TestLabels"
        return raw_dir / "TestVideos", label_dir if label_dir.exists() else None

    lowered = raw_dir.name.lower()
    if lowered == "trainvideos":
        label_dir = raw_dir.parent / "TrainLabels"
        return raw_dir, label_dir if label_dir.exists() else None
    if lowered == "testvideos":
        label_dir = raw_dir.parent / "TestLabels"
        return raw_dir, label_dir if label_dir.exists() else None

    return None, None


def find_track3_label_path(video_path: Path, video_dir: Path, label_dir: Path | None) -> Path:
    candidates: list[Path] = []
    if label_dir is not None:
        candidates.append(label_dir / f"{video_path.stem}.txt")

    collection_root = video_dir.parent
    numeric_suffix = video_path.stem.split("-")[-1]
    original_name = video_path.stem.replace("MultiUAV-", "UAV")
    candidates.append(collection_root / "original_label_file" / original_name / "gt" / "gt.txt")
    candidates.append(collection_root / "original_label_file" / f"UAV{numeric_suffix}" / "gt" / "gt.txt")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find a Track 3 label file for {video_path.name}. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


def list_train_sources(raw_dir: Path, modality: str) -> list[SequenceSource]:
    video_dir, label_dir = resolve_track3_video_layout(raw_dir)
    if video_dir is not None:
        return [
            SequenceSource(
                name=video_path.stem,
                media_path=video_path,
                annotation_path=find_track3_label_path(video_path, video_dir, label_dir),
                media_kind="video_file",
            )
            for video_path in list_video_files(video_dir)
        ]

    sources: list[SequenceSource] = []
    for sequence_dir in list_sequence_dirs(raw_dir):
        media_path, media_kind = resolve_sequence_media_source(sequence_dir, modality=modality)
        annotation_path = resolve_sequence_annotation_path(sequence_dir, modality=modality)
        if annotation_path is None and media_kind == "frame_dir":
            parsed = parse_mot_annotations(sequence_dir, task="auto")
            if parsed is None:
                raise FileNotFoundError(
                    f"No supported annotation file found in {sequence_dir} for modality={modality}. "
                    "Expected a modality-specific *_label.json or a MOT-style gt.txt file."
                )
            for candidate in (sequence_dir / "gt" / "gt.txt", sequence_dir / "gt.txt", sequence_dir / "annotations.txt"):
                if candidate.exists():
                    annotation_path = candidate
                    break
        sources.append(
            SequenceSource(
                name=sequence_dir.name,
                media_path=media_path,
                annotation_path=annotation_path,
                media_kind=media_kind,
            )
        )
    return sources


def list_test_sources(raw_dir: Path, modality: str) -> list[SequenceSource]:
    video_dir, _ = resolve_track3_video_layout(raw_dir)
    if video_dir is not None:
        return [
            SequenceSource(name=video_path.stem, media_path=video_path, annotation_path=None, media_kind="video_file")
            for video_path in list_video_files(video_dir)
        ]

    direct_videos = list_video_files(raw_dir)
    if direct_videos:
        return [
            SequenceSource(name=video_path.stem, media_path=video_path, annotation_path=None, media_kind="video_file")
            for video_path in direct_videos
        ]

    sources: list[SequenceSource] = []
    for sequence_dir in list_sequence_dirs(raw_dir):
        media_path, media_kind = resolve_sequence_media_source(sequence_dir, modality=modality)
        sources.append(
            SequenceSource(name=sequence_dir.name, media_path=media_path, annotation_path=None, media_kind=media_kind)
        )
    return sources


def build_output_stem(source_name: str, frame_stem: str) -> str:
    return f"{source_name}__{frame_stem}"


def convert_labeled_frame_sequence(
    source: SequenceSource,
    split: str,
    output_root: Path,
    task: str,
    copy_images: bool,
) -> SequenceSummary:
    if source.annotation_path is None:
        raise FileNotFoundError(f"Frame sequence is missing annotation path: {source.media_path}")

    frame_files = list_frame_files(source.media_path)
    if not frame_files:
        return SequenceSummary(source.name, 0, 0, 0, "no_frames")

    image = cv2.imread(str(frame_files[0]))
    if image is None:
        raise ValueError(f"Failed to read first frame in {source.media_path}")
    image_h, image_w = image.shape[:2]

    frame_boxes, annotation_mode = load_annotations_for_source(source, task=task)
    labeled_count = 0
    object_count = 0

    for frame_idx, frame_path in enumerate(frame_files):
        output_stem = build_output_stem(source.name, frame_path.stem)
        output_image_path = output_root / "images" / split / f"{output_stem}{frame_path.suffix.lower()}"
        output_label_path = output_root / "labels" / split / f"{output_stem}.txt"
        safe_link_or_copy(frame_path, output_image_path, copy_images=copy_images)

        frame_annotations = frame_boxes[frame_idx] if frame_idx < len(frame_boxes) else []
        yolo_lines = []
        for rect in frame_annotations:
            yolo_line = build_yolo_line(rect, image_w=image_w, image_h=image_h)
            if yolo_line is not None:
                yolo_lines.append(yolo_line)

        if yolo_lines:
            labeled_count += 1
            object_count += len(yolo_lines)
        output_label_path.write_text("".join(yolo_lines), encoding="utf-8")

    return SequenceSummary(
        name=source.name,
        frame_count=len(frame_files),
        labeled_count=labeled_count,
        object_count=object_count,
        annotation_mode=annotation_mode,
    )


def convert_labeled_video(
    source: SequenceSource,
    split: str,
    output_root: Path,
    task: str,
) -> SequenceSummary:
    if source.annotation_path is None:
        raise ValueError(f"Video source is missing annotation path: {source.media_path}")

    frame_boxes, annotation_mode = parse_mot_annotation_file(source.annotation_path, task=task)
    cap = cv2.VideoCapture(str(source.media_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {source.media_path}")

    frame_count = 0
    labeled_count = 0
    object_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            image_h, image_w = frame.shape[:2]
            output_stem = build_output_stem(source.name, f"{frame_count:06d}")
            output_image_path = output_root / "images" / split / f"{output_stem}.jpg"
            output_label_path = output_root / "labels" / split / f"{output_stem}.txt"

            if not output_image_path.exists() and not cv2.imwrite(str(output_image_path), frame):
                raise ValueError(f"Failed to write extracted frame: {output_image_path}")

            frame_annotations = frame_boxes[frame_count - 1] if frame_count <= len(frame_boxes) else []
            yolo_lines = []
            for rect in frame_annotations:
                yolo_line = build_yolo_line(rect, image_w=image_w, image_h=image_h)
                if yolo_line is not None:
                    yolo_lines.append(yolo_line)

            if yolo_lines:
                labeled_count += 1
                object_count += len(yolo_lines)
            output_label_path.write_text("".join(yolo_lines), encoding="utf-8")
    finally:
        cap.release()

    return SequenceSummary(
        name=source.name,
        frame_count=frame_count,
        labeled_count=labeled_count,
        object_count=object_count,
        annotation_mode=f"{annotation_mode}_video",
    )


def convert_labeled_source(
    source: SequenceSource,
    split: str,
    output_root: Path,
    task: str,
    copy_images: bool,
) -> SequenceSummary:
    if source.media_kind == "frame_dir":
        return convert_labeled_frame_sequence(source, split, output_root, task=task, copy_images=copy_images)
    if source.media_kind == "video_file":
        return convert_labeled_video(source, split, output_root, task=task)
    raise ValueError(f"Unsupported media kind: {source.media_kind}")


def convert_test_frame_sequence(source: SequenceSource, output_root: Path, copy_images: bool) -> int:
    frame_files = list_frame_files(source.media_path)
    for frame_path in frame_files:
        output_stem = build_output_stem(source.name, frame_path.stem)
        output_image_path = output_root / "images" / "test" / f"{output_stem}{frame_path.suffix.lower()}"
        safe_link_or_copy(frame_path, output_image_path, copy_images=copy_images)
    return len(frame_files)


def convert_test_video(source: SequenceSource, output_root: Path) -> int:
    cap = cv2.VideoCapture(str(source.media_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {source.media_path}")

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            output_stem = build_output_stem(source.name, f"{frame_count:06d}")
            output_image_path = output_root / "images" / "test" / f"{output_stem}.jpg"
            if not output_image_path.exists() and not cv2.imwrite(str(output_image_path), frame):
                raise ValueError(f"Failed to write extracted frame: {output_image_path}")
    finally:
        cap.release()

    return frame_count


def convert_test_source(source: SequenceSource, output_root: Path, copy_images: bool) -> int:
    if source.media_kind == "frame_dir":
        return convert_test_frame_sequence(source, output_root, copy_images=copy_images)
    if source.media_kind == "video_file":
        return convert_test_video(source, output_root)
    raise ValueError(f"Unsupported media kind: {source.media_kind}")


def summarize_split(name: str, summaries: list[SequenceSummary]) -> None:
    frame_count = sum(item.frame_count for item in summaries)
    labeled_frames = sum(item.labeled_count for item in summaries)
    object_count = sum(item.object_count for item in summaries)
    print(f"{name} frames: {frame_count}")
    print(f"{name} labeled frames: {labeled_frames}")
    print(f"{name} objects: {object_count}")


def main() -> None:
    args = parse_args()
    raw_train_dir = Path(args.raw_train_dir)
    output_root = Path(args.output_root)
    split_manifest_path = (
        Path(args.split_manifest) if args.split_manifest else default_split_manifest_path(raw_train_dir, seed=args.seed)
    )

    if not raw_train_dir.exists():
        raise FileNotFoundError(f"Raw Anti-UAV train directory not found: {raw_train_dir}")

    reset_split_dirs(output_root=output_root, clear_output=args.clear_output)
    train_sources = list_train_sources(raw_train_dir, modality=args.modality)
    train_sequences, val_sequences, manifest_status = split_sequences_with_manifest(
        train_sources,
        val_ratio=args.val_ratio,
        seed=args.seed,
        manifest_path=split_manifest_path,
        raw_train_dir=raw_train_dir,
        modality=args.modality,
    )

    print(f"Raw train dir: {raw_train_dir.resolve()}")
    print(f"YOLO output root: {output_root.resolve()}")
    print(f"Modality: {args.modality}")
    print(f"Task mode: {args.task}")
    print(f"Split manifest: {split_manifest_path.resolve()} ({manifest_status})")
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")

    train_summaries = [
        convert_labeled_source(seq, "train", output_root, task=args.task, copy_images=args.copy_images)
        for seq in train_sequences
    ]
    val_summaries = [
        convert_labeled_source(seq, "val", output_root, task=args.task, copy_images=args.copy_images)
        for seq in val_sequences
    ]

    print("\nConversion summary:")
    summarize_split("train", train_summaries)
    summarize_split("val", val_summaries)

    modes = sorted(set(item.annotation_mode for item in train_summaries + val_summaries))
    print(f"annotation modes: {', '.join(modes)}")
    media_kinds = sorted(set(seq.media_kind for seq in train_sequences + val_sequences))
    print(f"media kinds: {', '.join(media_kinds)}")

    if args.raw_test_dir:
        raw_test_dir = Path(args.raw_test_dir)
        if not raw_test_dir.exists():
            raise FileNotFoundError(f"Raw Anti-UAV test directory not found: {raw_test_dir}")
        test_sequence_dirs = list_test_sources(raw_test_dir, modality=args.modality)
        test_frames = sum(convert_test_source(seq, output_root, args.copy_images) for seq in test_sequence_dirs)
        print(f"test frames: {test_frames}")
        print(f"test sequences: {len(test_sequence_dirs)}")

    print("\n[OK] Anti-UAV data converted to YOLO format.")
    print(f"[OK] Images: {(output_root / 'images').resolve()}")
    print(f"[OK] Labels: {(output_root / 'labels').resolve()}")


if __name__ == "__main__":
    main()
