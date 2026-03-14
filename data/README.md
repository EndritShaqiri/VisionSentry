# Data Directory

Place your thermal UAV dataset in the following YOLO layout:

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

Each label file must match its image filename and contain YOLO boxes:

```text
class_id x_center y_center width height
```

All coordinates must be normalized to `[0, 1]`.

For this baseline, use a single class:
- `0 -> uav`
