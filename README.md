# VisionSentry

Production-ready baseline for **thermal/infrared UAV detection and tracking** using:
- **YOLOv12-style detector workflow** (Ultralytics API)
- **BoT-SORT tracker**
- **Optional distance estimation subsystem** under `distance_estimation/`
- Optional **ReID toggle** (off by default)
- **MOT-format export**: `frame,id,x,y,w,h,score,-1,-1,-1`

The repo currently has a solid **thermal/IR pipeline** under `src/`. Tracking is **inference-time tracking**, not a separately trained tracking model.

## What This Repo Does

## Demo Website

https://visionsentry-deployment.onrender.com/

## Demo Video

[Watch Demo Video (Google Drive)](https://drive.google.com/file/d/1D61qeDgZ1JTJ5EclSEITE89aEmOApSpu/view?usp=sharing)

[![VisionSentry Demo Video](https://drive.google.com/thumbnail?id=1D61qeDgZ1JTJ5EclSEITE89aEmOApSpu&sz=w1000)](https://drive.google.com/file/d/1D61qeDgZ1JTJ5EclSEITE89aEmOApSpu/view?usp=sharing)

## Run Web Visualization

Run the Gradio web app locally:

```bash
pip install -r requirements_app.txt
python drone_detection_app.py
```

Run with Docker:

```bash
docker build -t visionsentry:latest .
docker run --rm -p 7860:7860 --name visionsentry_local visionsentry:latest
```

Then open:

`http://127.0.0.1:7860`


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
  distance_estimation/
    configs/
      ranging.yaml
      camera_ir.yaml
      camera_rgb.yaml
    weights/
    estimator.py
    temporal.py
    train.py
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

## Dataset Format

Expected YOLO layout:

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

Label format:

```text
0 x_center y_center width height
```

Default dataset config is in [configs/dataset_thermal_uav.yaml](C:/Users/Thinkbook 14/VisionSentry/configs/dataset_thermal_uav.yaml):

```yaml
path: ./data/thermal_uav
train: images/train
val: images/val
test: images/test
names:
  0: uav
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt
```

Optional on SCC/Jupyter:

```bash
export PYTHONNOUSERSITE=1
```

## Prepare and Check Data

Verify dataset structure before training:

```bash
python -m src.utils.dataset_checks --data configs/dataset_thermal_uav.yaml
```

Strict mode:

```bash
python -m src.utils.dataset_checks --data configs/dataset_thermal_uav.yaml --strict
```

Convert raw Anti-UAV data:

```bash
python -m src.utils.prepare_anti_uav \
  --raw-train-dir data/raw_track1_2/train \
  --output-root data/thermal_uav \
  --val-ratio 0.2 \
  --clear-output

python -m src.utils.prepare_anti_uav \
  --raw-train-dir data/raw_track3/MultiUAV_Train \
  --output-root data/thermal_uav \
  --task multi \
  --val-ratio 0.2
```

Notes:
- splitting is by sequence, not by frame
- Track 1/2 frame folders and Track 3 video-plus-label data are both supported

## Train Detector

### Notebook workflow

Use [notebooks/train_detector.ipynb](C:/Users/Thinkbook 14/VisionSentry/notebooks/train_detector.ipynb).

Important:
- the notebook trains the **detector only**
- it calls the same `run_training(...)` function from [src/detection/train.py](C:/Users/Thinkbook 14/VisionSentry/src/detection/train.py)
- it does **not** train twice
- the notebook usually overrides the YAML config with a smaller smoke-test setup

Typical notebook smoke-test overrides:
- smaller image size
- fewer epochs
- auto batch sizing
- more workers
- disk cache

### CLI workflow

Config-driven:

```bash
python -m src.detection.train --config configs/train_detector.yaml
```

The default training config file is [configs/train_detector.yaml](C:/Users/Thinkbook 14/VisionSentry/configs/train_detector.yaml).

After training, the best checkpoint is saved at:

```text
runs/detect/<run_name>/weights/best.pt
```

## Validate Detector

```bash
python -m src.detection.validate \
  --weights runs/detect/<run_name>/weights/best.pt \
  --data configs/dataset_thermal_uav.yaml \
  --split val \
  --imgsz 960 \
  --batch 16 \
  --device 0 \
  --project runs/val \
  --name thermal_uav_val
```

Metrics are written to:

```text
runs/val/<run_name>/metrics.json
```

## Detector-Only Inference

This runs the trained detector on a video or frame folder. It does **not** do tracking.

Working example:

```bash
export PYTHONNOUSERSITE=1
python -m src.detection.infer \
  --weights runs/detect/yolo12n_thermal_uav_speedtest3/weights/best.pt \
  --source /projectnb/cs585/students/endrit01/VisionSentry/1.mp4 \
  --imgsz 640 \
  --conf 0.10 \
  --iou 0.5 \
  --device cpu \
  --save_video true \
  --save_frames true \
  --save_txt true \
  --project runs/predict \
  --name test_1mp4_detect_cpu
```

Outputs:
- `detected.mp4`
- `frames/`
- `detections.csv`

## Tracking

Tracking is run separately from detector training using BoT-SORT. In this repo, you do **not** train the tracker as a separate model. You train the detector, then run the tracker on top of detector predictions.

### Distance estimation on images or videos

Distance-estimation code lives entirely under `distance_estimation/`.

```bash
python -m src.detection.infer \
  --weights runs/detect/yolo12n_thermal_uav/weights/best.pt \
  --source data/sample.mp4 \
  --imgsz 960 \
  --conf 0.25 \
  --iou 0.5 \
  --device 0 \
  --ranging true \
  --ranging_config distance_estimation/configs/ranging.yaml \
  --camera_config distance_estimation/configs/camera_ir.yaml \
  --ranging_modality ir \
  --project runs/predict \
  --name thermal_detect_with_range
```

Additional outputs when ranging is enabled:
- `detections.csv` with distance, uncertainty, confidence, and range-bin columns
- `ranging_runtime.yaml` containing the effective distance-estimation config

## 7) BoT-SORT Tracking Inference

```bash
export PYTHONNOUSERSITE=1
python -m src.tracking.run_tracker \
  --weights runs/detect/yolo12n_thermal_uav_speedtest3/weights/best.pt \
  --source /projectnb/cs585/students/endrit01/VisionSentry/1.mp4 \
  --tracker configs/tracker_botsort.yaml \
  --imgsz 640 \
  --conf 0.10 \
  --iou 0.5 \
  --device cpu \
  --save_video true \
  --save_frames true \
  --project runs/track \
  --name test_1mp4_track_cpu
```

Outputs:
- tracked video: `tracked.mp4`
- optional annotated frames: `frames/`
- MOT file: `tracks_mot.txt` with rows:
  `frame,id,x,y,w,h,score,-1,-1,-1`
- `track_ranges.csv` when distance estimation is enabled

MOT rows are written as:

### Tracking + distance estimation

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
  --name thermal_track_with_range \
  --save_video true \
  --save_frames true \
  --ranging true \
  --ranging_config distance_estimation/configs/ranging.yaml \
  --camera_config distance_estimation/configs/camera_ir.yaml \
  --ranging_modality ir
```

The tracking pipeline keeps MOT export unchanged for compatibility and writes smoothed per-track distance estimates to `track_ranges.csv`.

## 8) ReID Toggle

Default tracker config is [configs/tracker_botsort.yaml](C:/Users/Thinkbook 14/VisionSentry/configs/tracker_botsort.yaml).

Current defaults:
- `with_reid: false`
- `gmc_method: None`
- `track_buffer: 60`
- `match_thresh: 0.90`

Enable ReID from CLI:

```bash
python -m src.tracking.run_tracker \
  --weights runs/detect/<run_name>/weights/best.pt \
  --source path/to/video.mp4 \
  --tracker configs/tracker_botsort.yaml \
  --with_reid true
```

## Recommended Workflow

1. Prepare raw data into `data/thermal_uav/`
2. Run dataset checks
3. Train detector
4. Validate detector
5. Run detector-only inference
6. Run tracking

## Current Status

- thermal detector/tracker pipeline is implemented under `src/`
- notebook workflow is supported
- tracking is working, but stable IDs depend heavily on detector quality
- the current saved `speedtest` weights were trained for only a short smoke test and are not final-quality weights

## Near-Term Roadmap

- stronger thermal detector training runs
- IR-only and RGB-only branch fusion using weighted box fusion
- future drone distance estimation
