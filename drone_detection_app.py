import gradio as gr
import cv2
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import tempfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from distance_estimation import DetectionInput, RangeEstimator


class DroneDetector:
    def __init__(self, model_path):
        """Initialize with thermal/IR model path"""
        self.model = YOLO(model_path)
        self.model_name = "Thermal/IR Drone Detector (YOLO12n)"
        self._range_estimator = None

    def _get_range_estimator(self):
        """Lazy-load the distance estimator (heuristic, CPU-friendly)."""
        if self._range_estimator is None:
            self._range_estimator = RangeEstimator.from_paths(
                config_path='distance_estimation/configs/ranging.yaml',
                modality='ir',
            )
        return self._range_estimator

    def _overlay_distance_above_label(self, frame_bgr, estimates):
        """Draw distance text above YOLO's class/ID label (which sits at the bbox top)."""
        annotated = frame_bgr
        label_offset_px = 18  # clears YOLO's default label band
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1
        for est in estimates:
            if est.distance_m is None:
                continue
            display_std = est.display_distance_std_m if est.display_distance_std_m is not None else est.distance_std_m
            text = f"{est.distance_m:.1f}m"
            if display_std is not None:
                text = f"{text} +/- {display_std:.1f}"
            if est.track_id is not None:
                text = f"ID {est.track_id} | {text}"
            x1 = int(max(0.0, est.x_center - (est.width / 2.0)))
            y_top = int(est.y_center - (est.height / 2.0))
            y_text = max(12, y_top - label_offset_px)
            color = (0, 255, 0) if not est.low_confidence else (0, 200, 255)
            cv2.putText(
                annotated, text, (x1 + 2, y_text),
                font, font_scale, color, thickness, cv2.LINE_AA,
            )
        return annotated

    def _detections_from_result(self, result, frame_index):
        """Convert an Ultralytics result into DetectionInput list for ranging."""
        if result.boxes is None or len(result.boxes) == 0:
            return []
        xywh = result.boxes.xywh.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.int().cpu().numpy()
        track_ids = None
        if getattr(result.boxes, "id", None) is not None:
            track_ids = result.boxes.id.int().cpu().numpy()
        detections = []
        for i, (box, score, cls_id) in enumerate(zip(xywh, confs, class_ids)):
            tid = int(track_ids[i]) if track_ids is not None else None
            detections.append(
                DetectionInput(
                    frame_index=frame_index,
                    class_id=int(cls_id),
                    score=float(score),
                    x_center=float(box[0]),
                    y_center=float(box[1]),
                    width=float(box[2]),
                    height=float(box[3]),
                    track_id=tid,
                )
            )
        return detections

    def process_image(self, image, enable_distance=False):
        """Process a single image with thermal/IR model"""
        if image is None:
            return None, "Please upload an image."

        # Ultralytics accepts RGB numpy arrays; run inference (CPU only for deployment)
        results = self.model(image, conf=0.25, iou=0.5, device='cpu')[0]

        # Get annotated image (BGR from Ultralytics) - font_size matches distance overlay
        annotated_bgr = results.plot(font_size=10, line_width=1)

        distance_summary = ""
        if enable_distance:
            estimator = self._get_range_estimator()
            detections = self._detections_from_result(results, frame_index=1)
            # Estimator expects BGR frames (matches annotated frame color order)
            frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
            estimates = estimator.estimate_detections(frame_bgr, detections, modality='ir')
            if estimates:
                annotated_bgr = self._overlay_distance_above_label(annotated_bgr, estimates)
                valid = [e.distance_m for e in estimates if e.distance_m is not None]
                if valid:
                    distance_summary = (
                        f"\n        **Distances (m)**: min {min(valid):.1f} "
                        f"/ mean {sum(valid)/len(valid):.1f} / max {max(valid):.1f}"
                    )

        # Get detection count
        count = len(results.boxes) if results.boxes is not None else 0

        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        stats = f"""
        ### Detection Results
        **Drones Detected**: {count}
        **Confidence Threshold**: 0.25
        **Model**: {self.model_name}
        **Distance Estimation**: {'ON' if enable_distance else 'OFF'}{distance_summary}
        """

        return img_rgb, stats
    
    def process_video(self, video_path, enable_distance=False, progress=gr.Progress()):
        """Process video with thermal/IR model"""
        # Check file size (100MB limit)
        import os
        if video_path and os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            max_size = 100 * 1024 * 1024  # 100MB in bytes
            if file_size > max_size:
                raise ValueError(f"Video file is too large ({file_size / 1024 / 1024:.1f}MB). Maximum allowed size is 100MB.")
        
        # Get video properties (via OpenCV probe only)
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Create temporary output directory (will be auto-deleted)
        temp_dir = tempfile.mkdtemp()
        output_dir = Path(temp_dir)
        
        # Generate unique filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"detected_{timestamp}.mp4"
        csv_path = output_dir / f"detections_{timestamp}.csv"
        
        # Video writer - use H264 codec for better browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Fallback to mp4v if avc1 fails
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Track detections
        total_detections = 0
        frame_count = 0

        # Track detections per frame for graphing
        detections_per_frame = []
        frame_numbers = []

        # Distance stats collectors
        estimator = self._get_range_estimator() if enable_distance else None
        all_distances = []
        unique_track_ids = set()

        # CSV writer setup
        csv_file = csv_path.open("w", encoding="utf-8", newline="")
        if enable_distance:
            csv_fields = [
                "frame", "track_id", "class_id", "x", "y", "w", "h", "score",
                "distance_m", "distance_std_m", "raw_distance_std_m", "display_distance_std_m",
                "distance_confidence", "quality_score", "range_bin", "low_confidence",
                "distance_min_m", "distance_max_m", "geometric_distance_m",
                "depth_distance_m", "used_fallback_camera", "notes",
            ]
        else:
            csv_fields = ["frame", "track_id", "class_id", "x", "y", "w", "h", "score"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()

        # Use YOLO tracking mode (BoT-SORT) so each drone gets a persistent ID
        stream = self.model.track(
            source=video_path,
            conf=0.25,
            iou=0.5,
            device='cpu',
            tracker='configs/tracker_botsort.yaml',
            persist=True,
            stream=True,
            verbose=False,
        )

        for results in stream:
            # Annotated frame from YOLO (includes class, confidence, and track id)
            annotated = results.plot(font_size=10, line_width=1)

            # Collect detections (with track ids) for ranging and stats
            detections = self._detections_from_result(results, frame_index=frame_count + 1)
            for det in detections:
                if det.track_id is not None:
                    unique_track_ids.add(det.track_id)

            # Optionally overlay distance estimates on top of tracked boxes
            estimates = []
            if estimator is not None and detections:
                estimates = estimator.estimate_detections(results.orig_img, detections, modality='ir')
                if estimates:
                    annotated = self._overlay_distance_above_label(annotated, estimates)
                    for est in estimates:
                        if est.distance_m is not None:
                            all_distances.append(est.distance_m)

            # Write per-detection CSV rows
            if enable_distance and estimates:
                for est in estimates:
                    row = est.as_csv_row()
                    csv_writer.writerow({k: row.get(k, "") for k in csv_fields})
            elif detections:
                for det in detections:
                    csv_writer.writerow({
                        "frame": det.frame_index,
                        "track_id": det.track_id if det.track_id is not None else "",
                        "class_id": det.class_id,
                        "x": round(det.x_center, 4),
                        "y": round(det.y_center, 4),
                        "w": round(det.width, 4),
                        "h": round(det.height, 4),
                        "score": round(det.score, 6),
                    })

            out.write(annotated)

            count = len(results.boxes) if results.boxes is not None else 0
            total_detections += count
            frame_numbers.append(frame_count)
            detections_per_frame.append(count)

            frame_count += 1
            if progress is not None and total_frames > 0:
                progress(min(frame_count / total_frames, 1.0),
                         desc=f"Processing frame {frame_count}/{total_frames}")

        out.release()
        csv_file.close()

        # Verify output file exists and has content
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Output video was not created properly")
        
        # Generate detection graph
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(frame_numbers, detections_per_frame, label='Drone Detections', linewidth=2, alpha=0.8, color='#FF6B6B')
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Number of Detections', fontsize=12)
        ax.set_title('Drone Detections Per Frame', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        # Calculate statistics
        avg_detections = total_detections / frame_count if frame_count > 0 else 0
        max_detections = max(detections_per_frame) if detections_per_frame else 0

        distance_block = ""
        if enable_distance:
            if all_distances:
                distance_block = (
                    "\n\n        **Distance Estimation** (heuristic, meters):\n"
                    f"        - Samples: {len(all_distances)}\n"
                    f"        - Min: {min(all_distances):.1f}\n"
                    f"        - Mean: {sum(all_distances) / len(all_distances):.1f}\n"
                    f"        - Max: {max(all_distances):.1f}"
                )
            else:
                distance_block = "\n\n        **Distance Estimation**: enabled, but no valid distances computed."

        stats = f"""
        ### Detection Statistics

        **{self.model_name}** (BoT-SORT tracking):
        - Total Detections: {total_detections}
        - Average per Frame: {avg_detections:.2f}
        - Peak Detections: {max_detections}
        - Unique Drone IDs: {len(unique_track_ids)}

        **Video Info**:
        - Total Frames: {frame_count}
        - FPS: {fps}
        - Duration: {frame_count/fps:.1f}s{distance_block}
        """
        
        # Load CSV for table preview
        try:
            csv_preview = pd.read_csv(csv_path)
        except Exception:
            csv_preview = pd.DataFrame()

        return (
            str(output_path),
            gr.update(value=str(csv_path), visible=True),
            gr.update(value=fig, visible=True),
            gr.update(value=csv_preview, visible=True),
            stats,
        )


def create_interface(model_path):
    """Create Gradio interface"""
    detector = DroneDetector(model_path)
    
    with gr.Blocks(title="VisionSentry") as demo:
        gr.Markdown("# VisionSentry")
        gr.Markdown("Thermal/IR drone detection, tracking, and distance estimation, powered by YOLO12n and BoT-SORT.")

        with gr.Tabs(selected="video"):
         with gr.Tab("Image Detection", id="image", visible=False):
            gr.Markdown("### Upload an image to detect drones")

            with gr.Row():
                image_input = gr.Image(label="Upload Image", type="numpy")

            with gr.Row():
                image_distance_toggle = gr.Checkbox(
                    label="Include Distance Estimation",
                    value=False,
                    info="Overlay estimated distance (meters) on each detection.",
                )

            with gr.Row():
                image_btn = gr.Button("Run Detection", variant="primary", size="lg")

            with gr.Row():
                image_output = gr.Image(label="Detection Results")

            image_stats = gr.Markdown()

            image_btn.click(
                fn=detector.process_image,
                inputs=[image_input, image_distance_toggle],
                outputs=[image_output, image_stats]
            )
        
         with gr.Tab("Video Detection", id="video"):
            gr.Markdown("""
            ### Upload a video to detect drones
            
            **File Size Limit**: Maximum 100MB per video
            
            **Note**: Processed videos are temporary and will be deleted when you close the browser or refresh the page.
            """)
            
            with gr.Row():
                video_input = gr.File(label="Upload Video File (Max 100MB)", file_types=["video"], type="filepath")

            with gr.Row():
                video_distance_toggle = gr.Checkbox(
                    label="Include Distance Estimation",
                    value=False,
                    info="Overlay estimated distance (meters) on each detection per frame.",
                )

            with gr.Accordion("How to read the overlay", open=False):
                gr.Markdown(
                    """
**Blue label** (drawn by YOLO at the top-left of each box): `uav 0.85 id:3`
- `uav`: detected class.
- `0.85`: detector confidence (range 0 to 1).
- `id:3`: persistent track ID assigned by BoT-SORT (same drone keeps the same ID across frames).

**Colored text above the blue label** (only when Include Distance Estimation is on): `ID 3 | 12.5m +/- 2.1`
- `ID 3`: same track ID as the blue label, shown again for convenience.
- `12.5m`: estimated distance to the drone in meters.
- `+/- 2.1`: one standard deviation of that distance estimate, also in meters. Smaller is more confident.
- If the estimate quality is too low for a precise number, the text shows a coarse range bin instead: `Close`, `Medium`, or `Distant`.

**Text color** (distance overlay):
- Green: the estimator considers the distance reasonably confident.
- Amber / orange: low-confidence estimate. Treat the number as a rough guide only.

**Bounding box**: drawn by the detector around every predicted drone location.

**Notes on distance accuracy**:
- Distances use a heuristic model (drone size prior + bounding-box geometry + camera intrinsics).
- Real camera intrinsics are not provided, so the system falls back to a generic IR camera profile. This inflates the uncertainty automatically.
- Very small boxes and low-confidence detections are penalized with larger `+/-` values.
                    """
                )

            with gr.Row():
                video_btn = gr.Button("Run Detection", variant="primary", size="lg")

            with gr.Column():
                video_output_annotated = gr.Video(label="Detection Results (Annotated)", autoplay=False)

            with gr.Row():
                csv_output = gr.File(label="Download Detections CSV", visible=False)

            video_stats = gr.Markdown()

            with gr.Row():
                graph_output = gr.Plot(label="Detections Per Frame", visible=False)

            with gr.Row():
                csv_table = gr.Dataframe(
                    label="Detections CSV Preview",
                    visible=False,
                    interactive=False,
                    wrap=True,
                    max_height=400,
                )

            video_btn.click(
                fn=detector.process_video,
                inputs=[video_input, video_distance_toggle],
                outputs=[video_output_annotated, csv_output, graph_output, csv_table, video_stats]
            )
        
        gr.Markdown("""
        ### Model Information
        - **Model**: Thermal/IR trained YOLO12n detector
        - **Confidence Threshold**: 0.25
        - **IoU Threshold**: 0.5
        - **Processing**: CPU-optimized for deployment
        
        ### Features
        - Real-time drone detection in thermal/IR imagery
        - Frame-by-frame analysis with detection graphs
        - Automatic video annotation
        - Detection statistics and metrics
        
        ### About
        This application uses a YOLO12n model trained on thermal/infrared UAV datasets
        to detect drones in various conditions. The model is optimized for CPU inference
        to ensure compatibility with cloud deployment platforms.
        """)

        gr.Markdown(
            """
            <hr style="margin-top: 2rem; margin-bottom: 1rem; border: none; border-top: 1px solid #e5e7eb;" />
            <div style="text-align: center; color: #64748b; font-size: 0.9rem; padding-bottom: 1rem;">
              <strong>VisionSentry Team</strong><br />
              Endrit Shaqiri &middot; James Njoroge &middot; Muhammad Raka Zuhdi &middot; Mofolaoluwarera Oladipo &middot; Jitarth Patel
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    import os

    # Model path - using thermal/IR model
    model_path = "weights/best.pt"

    # Respect Render's injected PORT, or GRADIO_SERVER_PORT, otherwise default to 7860
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    host = os.environ.get("HOST", os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))

    # Clean, professional theme (Gradio 6: pass theme to launch)
    theme = gr.themes.Default(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"],
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        body_background_fill="#f7f8fa",
        body_background_fill_dark="#0b1220",
        block_background_fill="white",
        block_background_fill_dark="#111827",
        block_border_width="1px",
        block_shadow="0 1px 2px rgba(15, 23, 42, 0.04)",
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="white",
    )

    # Create and launch interface
    demo = create_interface(model_path)
    demo.launch(server_name=host, server_port=port, theme=theme)
