# Data Directory

Place your thermal UAV dataset in the following YOLO layout:

```text
data/
  thermal_uav/
...
```

Each label file must match its image filename and contain one or more YOLO boxes:

```text
class_id x_center y_center width height
```

All coordinates must be normalized to `[0, 1]`.

For this baseline, use a single class:
- `0 -> uav`

Raw Anti-UAV data must be converted before training. The repo supports:
- Track 1/2 sequence folders with frames plus `IR_label.json`
- Track 3 `TrainVideos/` plus MOT-style `TrainLabels/*.txt`
