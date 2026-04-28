from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SplitReport:
    split_name: str
    image_count: int = 0
    label_count: int = 0
    missing_labels: int = 0
    missing_images: int = 0
    empty_labels: int = 0
    malformed_labels: int = 0
    class_counts: Counter[int] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset structure and labels.")
    parser.add_argument(
        "--data",
        type=str,
        default="configs/dataset_thermal_uav.yaml",
        help="Path to dataset YAML file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code on warnings like missing labels.",
    )
    return parser.parse_args()


def load_dataset_yaml(data_yaml_path: Path) -> dict:
    with data_yaml_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}
    return data_cfg


def resolve_dataset_root(data_cfg: dict, data_yaml_path: Path) -> Path:
    raw_path = data_cfg.get("path")
    if not raw_path:
        raise ValueError("Dataset YAML must contain the 'path' key.")

    root = Path(raw_path)
    if root.is_absolute():
        return root

    # Prefer project-root relative resolution (CLI usage), then YAML-relative fallback.
    cwd_candidate = (Path.cwd() / root).resolve()
    yaml_candidate = (data_yaml_path.parent / root).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return yaml_candidate


def gather_files(directory: Path, suffixes: Iterable[str]) -> list[Path]:
    return sorted([p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in suffixes])


def validate_label_file(
    label_path: Path,
    num_classes: int | None,
) -> tuple[int, int, Counter[int]]:
    malformed_lines = 0
    empty_file = 0
    class_counter: Counter[int] = Counter()

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return 0, 1, class_counter

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            malformed_lines += 1
            continue

        try:
            class_id_raw = float(parts[0])
            class_id = int(class_id_raw)
            if class_id_raw != class_id:
                malformed_lines += 1
                continue

            x, y, w, h = map(float, parts[1:])
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                malformed_lines += 1
                continue
            if w <= 0.0 or h <= 0.0:
                malformed_lines += 1
                continue
            if num_classes is not None and (class_id < 0 or class_id >= num_classes):
                malformed_lines += 1
                continue
        except ValueError:
            malformed_lines += 1
            continue

        class_counter[class_id] += 1

    return malformed_lines, empty_file, class_counter


def analyze_split(
    split_name: str,
    dataset_root: Path,
    image_rel_path: str,
    num_classes: int | None,
) -> SplitReport:
    image_dir = dataset_root / image_rel_path
    label_dir = dataset_root / str(Path(image_rel_path).as_posix().replace("images/", "labels/", 1))
    report = SplitReport(split_name=split_name)

    if not image_dir.exists():
        print(f"[ERROR] Missing image directory for split '{split_name}': {image_dir}")
        return report
    if not label_dir.exists():
        print(f"[ERROR] Missing label directory for split '{split_name}': {label_dir}")
        return report

    image_files = gather_files(image_dir, IMAGE_EXTENSIONS)
    label_files = gather_files(label_dir, {".txt"})
    report.image_count = len(image_files)
    report.label_count = len(label_files)

    image_map = {(p.relative_to(image_dir)).with_suffix(""): p for p in image_files}
    label_map = {(p.relative_to(label_dir)).with_suffix(""): p for p in label_files}

    for image_rel_no_suffix in image_map:
        if image_rel_no_suffix not in label_map:
            report.missing_labels += 1

    for label_rel_no_suffix in label_map:
        if label_rel_no_suffix not in image_map:
            report.missing_images += 1

    for label_path in label_files:
        malformed_lines, empty_file, class_counter = validate_label_file(label_path, num_classes=num_classes)
        report.malformed_labels += malformed_lines
        report.empty_labels += empty_file
        report.class_counts.update(class_counter)

    return report


def print_report(reports: list[SplitReport]) -> tuple[int, Counter[int]]:
    total_issues = 0
    total_classes: Counter[int] = Counter()

    print("\n=== Dataset Verification Summary ===")
    for rep in reports:
        total_classes.update(rep.class_counts)
        issues = rep.missing_labels + rep.missing_images + rep.empty_labels + rep.malformed_labels
        total_issues += issues
        print(f"\n[{rep.split_name}]")
        print(f"  images:           {rep.image_count}")
        print(f"  labels:           {rep.label_count}")
        print(f"  missing labels:   {rep.missing_labels}")
        print(f"  missing images:   {rep.missing_images}")
        print(f"  empty label files:{rep.empty_labels}")
        print(f"  malformed labels: {rep.malformed_labels}")

    print("\nClass statistics:")
    if total_classes:
        for cls_id, count in sorted(total_classes.items()):
            print(f"  class {cls_id}: {count} boxes")
    else:
        print("  No valid annotations found.")

    return total_issues, total_classes


def main() -> None:
    args = parse_args()
    data_yaml_path = Path(args.data).resolve()
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")

    data_cfg = load_dataset_yaml(data_yaml_path)
    dataset_root = resolve_dataset_root(data_cfg, data_yaml_path)
    names = data_cfg.get("names", {})
    num_classes = len(names) if isinstance(names, dict) else (len(names) if isinstance(names, list) else None)

    print(f"Dataset YAML: {data_yaml_path}")
    print(f"Dataset root: {dataset_root}")
    if num_classes is not None:
        print(f"Configured classes: {num_classes}")

    split_keys = [k for k in ("train", "val", "test") if k in data_cfg]
    if not split_keys:
        raise ValueError("Dataset YAML must define at least one split key among: train, val, test")

    reports = [analyze_split(split, dataset_root, data_cfg[split], num_classes=num_classes) for split in split_keys]
    total_issues, _ = print_report(reports)

    if total_issues > 0:
        warning_message = (
            f"\n[WARN] Dataset verification found {total_issues} issue(s). "
            "Review warnings before training."
        )
        print(warning_message)
        if args.strict:
            raise SystemExit(1)

    print("\n[OK] Dataset verification completed.")


if __name__ == "__main__":
    main()
