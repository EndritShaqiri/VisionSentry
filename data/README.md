# Data Directory

Place your UAV datasets in the following YOLO layouts:

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
  rgb_uav/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
```

Each label file must match its image filename and contain one or more YOLO boxes:

```text
class_id x_center y_center width height
```

All coordinates must be normalized to `[0, 1]`.

For this baseline, use a single class for both modalities:
- `0 -> uav`

Raw Anti-UAV data must be converted before training. The repo supports:
- Track 1/2 sequence folders with frames plus `IR_label.json`
- Track 1/2 RGB sequence folders with `RGB_label.json` or `visible_label.json`
- Track 3 `TrainVideos/` plus MOT-style `TrainLabels/*.txt`

Recommended workflow:
- convert infrared data into `data/thermal_uav/` with `--modality ir`
- convert RGB data into `data/rgb_uav/` with `--modality rgb`
- reuse the same `--split-manifest` path across both runs so train/val splits stay aligned
