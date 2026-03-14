# VisionSentry: Thermal UAV Detection + Tracking Baseline

Clean, modular, production-ready baseline for **thermal/infrared UAV detection and tracking** using:
- **YOLOv12-style detector workflow** (Ultralytics API)
- **BoT-SORT tracker**
- Optional **ReID toggle** (off by default)
- **MOT-format export**: `frame,id,x,y,w,h,score,-1,-1,-1`

This repository is intentionally general-purpose and not tied to any competition-specific conventions.

## 1) Project Structure

```text
project_root/
  configs/
    dataset_thermal_uav.yaml
    tracker_botsort.yaml
    train_detector.yaml
  data/
    README.md
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

Label format (single class `uav`):

```text
0 x_center y_center width height
```

Coordinates are normalized `[0, 1]`.

### Dataset Config
Edit:
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

### Verify Dataset Before Training

```bash
python -m src.utils.dataset_checks --data configs/dataset_thermal_uav.yaml
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

### Convert Anti-UAV Track 2 Raw Data

If you extracted the official release into `data/raw/train`, convert it into the YOLO layout with:

```bash
python -m src.utils.prepare_anti_uav_track2 \
  --raw-train-dir data/raw/train \
  --output-root data/thermal_uav \
  --val-ratio 0.2 \
  --clear-output
```

Notes:
- The converter splits by sequence, not by frame, to avoid train/val leakage.
- Official `track2_test` can remain under `data/raw/track2_test/` and be used later as an inference source.

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
```

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

## 8) ReID Toggle

Default is disabled in:
- `configs/tracker_botsort.yaml` (`with_reid: false`)

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
2. Install dependencies.
3. Upload/mount dataset into `data/thermal_uav/`.
4. Run dataset check.
5. Train detector.
6. Validate detector.
7. Run detection + tracking and export MOT results.

## 10) First Things To Run

If you just cloned/opened the repo:

```bash
pip install -r requirements.txt
python -m src.utils.dataset_checks --data configs/dataset_thermal_uav.yaml
python -m src.detection.train --config configs/train_detector.yaml
```

## 11) Which File To Edit First For Your Dataset

Edit this first:
- `configs/dataset_thermal_uav.yaml`

Then place your images/labels under:
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
   - add a new loader/module under `src/detection/` for dual-stream input
   - add a fusion model config and fusion training entry point
2. Higher resolutions:
   - update `imgsz` in config/CLI and tune batch size
3. Thermal enhancement:
   - add preprocessing pipeline under `src/utils/` (CLAHE, denoising, super-resolution)
   - call it in `src/detection/infer.py` and training dataloader path
4. Acoustic fusion:
   - add a feature encoder branch and late fusion tracker association module under `src/tracking/`

The current design keeps interfaces stable so these additions can be layered on top of the existing detector + tracker pipeline.
