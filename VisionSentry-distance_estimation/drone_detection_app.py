import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import tempfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class DroneDetector:
    def __init__(self, model_path):
        """Initialize with thermal/IR model path"""
        self.model = YOLO(model_path)
        self.model_name = "Thermal/IR Drone Detector"
    
    def process_image(self, image):
        """Process a single image with thermal/IR model"""
        # Run inference (CPU only for deployment)
        results = self.model(image, conf=0.25, iou=0.5, device='cpu')[0]
        
        # Get annotated image
        img = results.plot()
        
        # Get detection count
        count = len(results.boxes) if results.boxes is not None else 0
        
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        stats = f"""
        ### Detection Results
        **Drones Detected**: {count}
        **Confidence Threshold**: 0.25
        **Model**: {self.model_name}
        """
        
        return img_rgb, stats
    
    def process_video(self, video_path, progress=gr.Progress()):
        """Process video with thermal/IR model"""
        # Check file size (100MB limit)
        import os
        if video_path and os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            max_size = 100 * 1024 * 1024  # 100MB in bytes
            if file_size > max_size:
                raise ValueError(f"Video file is too large ({file_size / 1024 / 1024:.1f}MB). Maximum allowed size is 100MB.")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create temporary output directory (will be auto-deleted)
        temp_dir = tempfile.mkdtemp()
        output_dir = Path(temp_dir)
        
        # Generate unique filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"detected_{timestamp}.mp4"
        
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
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference (CPU only for deployment)
            results = self.model(frame, conf=0.25, iou=0.5, device='cpu', verbose=False)[0]
            
            # Get annotated frame
            annotated = results.plot()
            
            # Write frame
            out.write(annotated)
            
            # Count detections for this frame
            count = len(results.boxes) if results.boxes is not None else 0
            
            total_detections += count
            
            # Track per-frame data
            frame_numbers.append(frame_count)
            detections_per_frame.append(count)
            
            frame_count += 1
            if progress is not None:
                progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")
        
        cap.release()
        out.release()
        
        # Verify output file exists and has content
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Output video was not created properly")
        
        # Generate detection graph
        graph_path = output_dir / f"detections_graph_{timestamp}.png"
        plt.figure(figsize=(12, 6))
        plt.plot(frame_numbers, detections_per_frame, label='Drone Detections', linewidth=2, alpha=0.8, color='#FF6B6B')
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Number of Detections', fontsize=12)
        plt.title('Drone Detections Per Frame', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate statistics
        avg_detections = total_detections / frame_count if frame_count > 0 else 0
        max_detections = max(detections_per_frame) if detections_per_frame else 0
        
        stats = f"""
        ### Detection Statistics
        
        **{self.model_name}**:
        - Total Detections: {total_detections}
        - Average per Frame: {avg_detections:.2f}
        - Peak Detections: {max_detections}
        
        **Video Info**:
        - Total Frames: {frame_count}
        - FPS: {fps}
        - Duration: {frame_count/fps:.1f}s
        """
        
        return str(output_path), str(graph_path), stats


def create_interface(model_path):
    """Create Gradio interface"""
    detector = DroneDetector(model_path)
    
    with gr.Blocks(title="IR Drone Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚁 Thermal/IR Drone Detection")
        gr.Markdown("Detect drones in thermal/infrared images and videos using YOLOv11")
        
        with gr.Tab("Image Detection"):
            gr.Markdown("### Upload an image to detect drones")
            
            with gr.Row():
                image_input = gr.Image(label="Upload Image", type="numpy")
            
            with gr.Row():
                image_btn = gr.Button("Run Detection", variant="primary", size="lg")
            
            with gr.Row():
                image_output = gr.Image(label="Detection Results")
            
            image_stats = gr.Markdown()
            
            image_btn.click(
                fn=detector.process_image,
                inputs=[image_input],
                outputs=[image_output, image_stats]
            )
        
        with gr.Tab("Video Detection"):
            gr.Markdown("""
            ### Upload a video to detect drones
            
            **File Size Limit**: Maximum 100MB per video
            
            **Note**: Processed videos are temporary and will be deleted when you close the browser or refresh the page.
            """)
            
            with gr.Row():
                video_input = gr.Video(label="Upload Video (Max 100MB)")
            
            with gr.Row():
                video_btn = gr.Button("Run Detection", variant="primary", size="lg")
            
            with gr.Row():
                video_output = gr.Video(label="Detection Results", autoplay=False)
            
            video_stats = gr.Markdown()
            
            with gr.Row():
                graph_output = gr.Image(label="Detections Per Frame", type="filepath")
            
            video_btn.click(
                fn=detector.process_video,
                inputs=[video_input],
                outputs=[video_output, graph_output, video_stats]
            )
        
        gr.Markdown("""
        ### Model Information
        - **Model**: Thermal/IR trained YOLOv11 detector
        - **Confidence Threshold**: 0.25
        - **IoU Threshold**: 0.5
        - **Processing**: CPU-optimized for deployment
        
        ### Features
        - Real-time drone detection in thermal/IR imagery
        - Frame-by-frame analysis with detection graphs
        - Automatic video annotation
        - Detection statistics and metrics
        
        ### About
        This application uses a YOLOv11 model trained on thermal/infrared UAV datasets
        to detect drones in various conditions. The model is optimized for CPU inference
        to ensure compatibility with cloud deployment platforms.
        """)
    
    return demo


if __name__ == "__main__":
    # Model path - using thermal/IR model
    model_path = "weights/best.pt"
    
    # Create and launch interface
    demo = create_interface(model_path)
    demo.launch(server_name="0.0.0.0", server_port=7860)
