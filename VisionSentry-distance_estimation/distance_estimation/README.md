# Distance Estimation

This folder contains the standalone ranging subsystem for VisionSentry.

It is intentionally isolated from detector training so it can be developed and iterated independently for:
- IR-only inference
- RGB-only inference
- detector outputs on still images
- tracked outputs on videos
- future RGB/IR fusion

## Layout

```text
distance_estimation/
  configs/
    ranging.yaml
    camera_ir.yaml
    camera_rgb.yaml
  weights/
    .gitkeep
  __init__.py
  backbones.py
  config.py
  estimator.py
  geometry.py
  models.py
  temporal.py
  train.py
  types.py
```

## What Works Now

- Heuristic metric-distance estimation from detections with uncertainty
- Optional plug-in depth backbone interface
- Tracking-aware offline smoothing for per-track video distance
- CSV export and overlay integration from the main inference/tracking entrypoints
- Lightweight `DroneRangeHead` training CLI for supervised feature CSVs

## Default Runtime

The shipped default is a hybrid heuristic estimator:
- camera fallback intrinsics from config
- bbox-to-distance geometry from drone size priors
- optional dense depth cue if a custom depth backbone is plugged in
- uncertainty inflation for tiny boxes, low detector confidence, and unknown camera intrinsics

To train a learned range head later:

```bash
python -m distance_estimation.train \
  --features_csv path/to/features.csv \
  --output distance_estimation/weights/range_head.pt
```
