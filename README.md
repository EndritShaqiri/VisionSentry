# VisionSentry: Thermal + RGB UAV Detection + Tracking Baseline

Clean, modular, production-ready baseline for **thermal/infrared and RGB UAV detection and tracking** using:
- **YOLOv12-style detector workflow** (Ultralytics API)
- **BoT-SORT tracker**
- Optional **ReID toggle** (off by default)
- **MOT-format export**: `frame,id,x,y,w,h,score,-1,-1,-1`

This repository supports both:
- **single-UAV experiments** such as Anti-UAV Track 2
- **multi-UAV experiments** when the raw annotations provide multiple boxes per frame

The runtime pipeline is intentionally general-purpose and not tied to any competition-specific conventions.

## 1) Project Structure

```text
project_root/
  configs/
    dataset_rgb_uav.yaml
    dataset_thermal_uav.yaml
    tracker_botsort.yaml
    train_detector.yaml
    train_detector_rgb.yaml
  data/
    README.md
    rgb_uav/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
    thermal_uav/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
  notebooks/
    train_detector.ipynb
    infer_and_track.ipynb
  runs/
  src/
    detection/
      train.py
      validate.py
      infer.py
    tracking/
      run_tracker.py
    utils/
      dataset_checks.py
      paths.py
      visualization.py
  requirements.txt
  README.md
```

## 2) Dataset Layout (YOLO)

Expected dataset layout:

```text
data/
  rgb_uav/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
  thermal_uav/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
```

Label format (single class `uav`, one or more boxes per frame):

```text
0 x_center y_center width height
```

Coordinates are normalized `[0, 1]`.

### Dataset Config
Edit:
- `configs/dataset_rgb_uav.yaml`
- `configs/dataset_thermal_uav.yaml`

Default:

```yaml
path: ./data/thermal_uav
train: images/train
val: images/val
test: images/test
names:
  0: uav
```

RGB default:

```yaml
path: ./data/rgb_uav
train: images/train
val: images/val
test: images/test
names:
  0: uav
```

### Verify Dataset Before Training

```bash
python -m src.utils.dataset_checks --data configs/dataset_thermal_uav.yaml
python -m src.utils.dataset_checks --data configs/dataset_rgb_uav.yaml
```

Strict mode (non-zero exit if issues are found):

```bash
python -m src.utils.dataset_checks --data configs/dataset_thermal_uav.yaml --strict
```

Checks include:
- image/label count consistency
- malformed label lines
- empty label files
- class statistics

### Convert Anti-UAV Raw Data

If you extracted the official release into `data/raw/train`, convert the infrared split into the YOLO layout with:

```bash
python -m src.utils.prepare_anti_uav \
  --raw-train-dir data/raw/train \
  --output-root data/thermal_uav \
  --modality ir \
  --val-ratio 0.2 \
  --clear-output
```

To build the RGB dataset from the same raw Anti-UAV release and keep the exact same train/val partition, reuse a shared split manifest:

```bash
python -m src.utils.prepare_anti_uav \
  --raw-train-dir data/raw/train \
  --output-root data/rgb_uav \
  --modality rgb \
  --split-manifest data/raw/train_split_seed42.json \
  --val-ratio 0.2 \
  --clear-output
```

Notes:
- The converter splits by sequence, not by frame, to avoid train/val leakage.
- The converter now supports `--modality ir|rgb` for frame-based Anti-UAV layouts.
- Reuse the same `--split-manifest` across infrared and RGB conversions to keep metrics comparable.
- It auto-detects common Anti-UAV annotation layouts for **single-target** and **multi-target** training data.
- It also auto-detects the 4th Anti-UAV Track 3 layout with `TrainVideos/` plus `TrainLabels/` and decodes videos directly into frame-level YOLO samples.
- Official `track2_test` or `track3_test` can remain under `data/raw/...` and be used later as an inference source.

If you have **Track 1/2 raw frame folders** plus **Track 3 raw videos**, build the combined training set by running the converter twice into the same output root:

```bash
python -m src.utils.prepare_anti_uav \
  --raw-train-dir data/raw_track1_2/train \
  --output-root data/thermal_uav \
  --modality ir \
  --val-ratio 0.2 \
  --clear-output

python -m src.utils.prepare_anti_uav \
  --raw-train-dir data/raw_track3/MultiUAV_Train \
  --output-root data/thermal_uav \
  --task multi \
  --modality ir \
  --val-ratio 0.2
```

For Track 3, you can also call:

```bash
python -m src.utils.prepare_anti_uav_track3 \
  --raw-train-dir data/raw_track3/MultiUAV_Train \
  --output-root data/thermal_uav \
  --task multi \
  --modality ir \
  --val-ratio 0.2
```

Prepare test images as well:

```bash
python -m src.utils.prepare_anti_uav \
  --raw-train-dir data/raw/train \
  --raw-test-dir data/raw/track2_test \
  --output-root data/thermal_uav \
  --modality ir \
  --val-ratio 0.2 \
  --clear-output
```

## 3) Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## 4) Train Detector (YOLOv12-style)

### Option A: config-driven

```bash
python -m src.detection.train --config configs/train_detector.yaml
python -m src.detection.train --config configs/train_detector_rgb.yaml
```

The same training entry point is now callable directly from notebooks:

```python
from pathlib import Path
from src.detection.train import load_yaml, run_training
from src.utils.project import find_project_root

project_root = find_project_root()
cfg = load_yaml(project_root / "configs" / "train_detector.yaml")
cfg.update({"device": "auto", "epochs": 1, "imgsz": 640, "batch": 8})
save_dir = run_training(cfg, project_root=project_root)
```

The training entry point now also forwards the notebook-level Ultralytics options used in practice:
- `amp`
- `verbose`
- `seed`
- `deterministic`

### Option B: CLI overrides

```bash
python -m src.detection.train \
  --model yolo12n.pt \
  --data configs/dataset_thermal_uav.yaml \
  --imgsz 960 \
  --batch 16 \
  --epochs 100 \
  --device 0 \
  --workers 8 \
  --project runs/detect \
  --name yolo12n_thermal_uav
```

RGB parity run:

```bash
python -m src.detection.train \
  --model yolo12n.pt \
  --data configs/dataset_rgb_uav.yaml \
  --imgsz 960 \
  --batch 16 \
  --epochs 100 \
  --device 0 \
  --workers 8 \
  --project runs/detect \
  --name yolo12n_rgb_uav \
  --amp true \
  --verbose true
```

Best checkpoint is saved to:

```text
runs/detect/<run_name>/weights/best.pt
```

## 5) Validate Detector

```bash
python -m src.detection.validate \
  --weights runs/detect/yolo12n_thermal_uav/weights/best.pt \
  --data configs/dataset_thermal_uav.yaml \
  --split val \
  --imgsz 960 \
  --batch 16 \
  --device 0 \
  --project runs/val \
  --name thermal_uav_val
```

Metrics are saved in:
- `runs/val/<run_name>/metrics.json`

## 6) Detector-Only Inference

Works on a video file or folder of frames:

```bash
python -m src.detection.infer \
  --weights runs/detect/yolo12n_thermal_uav/weights/best.pt \
  --source data/sample.mp4 \
  --imgsz 960 \
  --conf 0.25 \
  --iou 0.5 \
  --device 0 \
  --save_video true \
  --save_frames true \
  --save_txt true \
  --project runs/predict \
  --name thermal_detect
```

Outputs:
- `detected.mp4`
- optional annotated frames (`frames/`)
- optional `detections.csv`

For RGB inference, swap in the RGB checkpoint path, for example:
- `runs/detect/yolo12n_rgb_uav/weights/best.pt`

## 7) BoT-SORT Tracking Inference

```bash
python -m src.tracking.run_tracker \
  --weights runs/detect/yolo12n_thermal_uav/weights/best.pt \
  --source data/sample.mp4 \
  --tracker configs/tracker_botsort.yaml \
  --conf 0.25 \
  --iou 0.5 \
  --imgsz 960 \
  --device 0 \
  --project runs/track \
  --name thermal_track \
  --save_video true \
  --save_frames true
```

Outputs:
- tracked video: `tracked.mp4`
- optional annotated frames: `frames/`
- MOT file: `tracks_mot.txt` with rows:
  `frame,id,x,y,w,h,score,-1,-1,-1`

The same tracker entry point works for:
- single-UAV sequences
- multi-UAV sequences with multiple detections per frame
- RGB checkpoints as well as thermal checkpoints

## 8) ReID Toggle

Default is disabled in:
- `configs/tracker_botsort.yaml` (`with_reid: false`)

The default tracker config is tuned for thermal UAV footage:
- lower thresholds to reduce track resets
- larger `track_buffer` to survive short detection misses
- `gmc_method: None` to avoid unstable sparse optical flow on low-texture thermal scenes

RGB checkpoints work with the same tracker entry point, but phase 1 keeps tracker tuning unchanged. Treat RGB tracking output as compatible qualitative inference until you decide to tune a separate RGB tracker config.

Enable from CLI without editing config:

```bash
python -m src.tracking.run_tracker \
  --weights runs/detect/yolo12n_thermal_uav/weights/best.pt \
  --source data/sample.mp4 \
  --tracker configs/tracker_botsort.yaml \
  --with_reid true
```

A runtime tracker config is written to the run folder for reproducibility.

## 9) Google Colab Workflow

Use the provided notebooks:
- `notebooks/train_detector.ipynb`
- `notebooks/infer_and_track.ipynb`

Typical flow in Colab:
1. Open repo in Colab runtime.
2. Set `MODALITY` to `ir` or `rgb`.
3. Optionally let the notebook download the public Anti-UAV-RGBT archive for RGB experiments.
4. Convert raw Anti-UAV data into `data/thermal_uav/` or `data/rgb_uav/`.
5. Run dataset checks.
6. Run a smoke test.
7. Switch to parity settings and rerun training.
8. Validate detector and compare metrics against the checked-in thermal baseline.
9. Run detection + tracking and export MOT results.

On SCC Jupyter:
1. Start the notebook from the same conda environment you prepared for training.
2. Open `notebooks/train_detector.ipynb`.
3. Run the CUDA probe cell before training.
4. Start with the notebook smoke-test config.
5. Increase epochs/image size/batch only after the smoke test succeeds.

## 10) First Things To Run

If you just cloned/opened the repo:

```bash
pip install -r requirements.txt
python -m src.utils.dataset_checks --data configs/dataset_thermal_uav.yaml
python -m src.detection.train --config configs/train_detector.yaml
```

For RGB:

```bash
python -m src.utils.prepare_anti_uav \
  --raw-train-dir data/raw/train \
  --output-root data/rgb_uav \
  --modality rgb \
  --split-manifest data/raw/train_split_seed42.json
python -m src.utils.dataset_checks --data configs/dataset_rgb_uav.yaml
python -m src.detection.train --config configs/train_detector_rgb.yaml
```

## 11) Which File To Edit First For Your Dataset

Edit this first:
- `configs/dataset_rgb_uav.yaml`
- `configs/dataset_thermal_uav.yaml`

Then place your images/labels under:
- `data/rgb_uav/images/*`
- `data/rgb_uav/labels/*`
- `data/thermal_uav/images/*`
- `data/thermal_uav/labels/*`

## 12) Where `best.pt` Goes

After training, `best.pt` is automatically created at:
- `runs/detect/<run_name>/weights/best.pt`

For inference/tracking, pass that path to:
- `--weights`

## 13) Extension Roadmap

This codebase is structured to extend without breaking current workflows:

1. RGB + Thermal fusion:
   - keep the new modality-aware dataset preparation path
   - add a dual-stream model only after detector parity is established
2. Higher resolutions:
   - update `imgsz` in config/CLI and tune batch size
3. Thermal enhancement:
   - add preprocessing pipeline under `src/utils/` (CLAHE, denoising, super-resolution)
   - call it in `src/detection/infer.py` and training dataloader path
4. Acoustic fusion:
   - add a feature encoder branch and late fusion tracker association module under `src/tracking/`

The current design keeps interfaces stable so these additions can be layered on top of the existing detector + tracker pipeline.
