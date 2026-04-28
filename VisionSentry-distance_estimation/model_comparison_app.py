import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import tempfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class ModelComparison:
    def __init__(self, model1_path, model2_path):
        """Initialize with two model paths"""
        self.model1 = YOLO(model1_path)
        self.model2 = YOLO(model2_path)
        self.model1_name = Path(model1_path).parent.parent.name if 'runs' in model1_path else "Thermal Model"
        self.model2_name = Path(model2_path).parent.parent.name if 'runs' in model2_path else "RGB Model"
    
    def process_image(self, image):
        """Process a single image with both models"""
        # Run inference with both models (CPU only for deployment)
        results1 = self.model1(image, conf=0.25, iou=0.5, device='cpu')[0]
        results2 = self.model2(image, conf=0.25, iou=0.5, device='cpu')[0]
        
        # Get annotated images
        img1 = results1.plot()
        img2 = results2.plot()
        
        # Get detection counts
        count1 = len(results1.boxes) if results1.boxes is not None else 0
        count2 = len(results2.boxes) if results2.boxes is not None else 0
        
        # Convert BGR to RGB for display
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        stats = f"""
        **Model 1 ({self.model1_name})**: {count1} detections
        **Model 2 ({self.model2_name})**: {count2} detections
        """
        
        return img1_rgb, img2_rgb, stats
    
    def process_video(self, video_path, progress=gr.Progress()):
        """Process video with both models"""
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
        
        # Generate unique filenames with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output1_path = output_dir / f"model1_{timestamp}.mp4"
        output2_path = output_dir / f"model2_{timestamp}.mp4"
        
        # Video writers - use H264 codec for better browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out1 = cv2.VideoWriter(str(output1_path), fourcc, fps, (width, height))
        out2 = cv2.VideoWriter(str(output2_path), fourcc, fps, (width, height))
        
        # Check if video writers opened successfully
        if not out1.isOpened() or not out2.isOpened():
            # Fallback to mp4v if avc1 fails
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out1 = cv2.VideoWriter(str(output1_path), fourcc, fps, (width, height))
            out2 = cv2.VideoWriter(str(output2_path), fourcc, fps, (width, height))
        
        total_detections1 = 0
        total_detections2 = 0
        frame_count = 0
        
        # Track detections per frame for graphing
        detections_per_frame1 = []
        detections_per_frame2 = []
        frame_numbers = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference (CPU only for deployment)
            results1 = self.model1(frame, conf=0.25, iou=0.5, device='cpu', verbose=False)[0]
            results2 = self.model2(frame, conf=0.25, iou=0.5, device='cpu', verbose=False)[0]
            
            # Get annotated frames
            annotated1 = results1.plot()
            annotated2 = results2.plot()
            
            # Write frames
            out1.write(annotated1)
            out2.write(annotated2)
            
            # Count detections for this frame
            count1 = len(results1.boxes) if results1.boxes is not None else 0
            count2 = len(results2.boxes) if results2.boxes is not None else 0
            
            total_detections1 += count1
            total_detections2 += count2
            
            # Track per-frame data
            frame_numbers.append(frame_count)
            detections_per_frame1.append(count1)
            detections_per_frame2.append(count2)
            
            frame_count += 1
            if progress is not None:
                progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")
        
        cap.release()
        out1.release()
        out2.release()
        
        # Verify output files exist and have content
        if not output1_path.exists() or output1_path.stat().st_size == 0:
            raise RuntimeError("Model 1 output video was not created properly")
        if not output2_path.exists() or output2_path.stat().st_size == 0:
            raise RuntimeError("Model 2 output video was not created properly")
        
        # Generate comparison graph
        graph_path = output_dir / f"detections_graph_{timestamp}.png"
        plt.figure(figsize=(12, 6))
        plt.plot(frame_numbers, detections_per_frame1, label=f'Model 1: {self.model1_name}', linewidth=2, alpha=0.8)
        plt.plot(frame_numbers, detections_per_frame2, label=f'Model 2: {self.model2_name}', linewidth=2, alpha=0.8)
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Number of Detections', fontsize=12)
        plt.title('Detections Per Frame Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        stats = f"""
        **Processed {frame_count} frames**
        
        **Model 1 ({self.model1_name})**: {total_detections1} total detections ({total_detections1/frame_count:.2f} per frame)
        **Model 2 ({self.model2_name})**: {total_detections2} total detections ({total_detections2/frame_count:.2f} per frame)
        
        **Output files created:**
        - Model 1: {output1_path.stat().st_size / 1024 / 1024:.2f} MB
        - Model 2: {output2_path.stat().st_size / 1024 / 1024:.2f} MB
        """
        
        return str(output1_path), str(output2_path), str(graph_path), stats


def create_interface(model1_path, model2_path):
    """Create Gradio interface"""
    comparator = ModelComparison(model1_path, model2_path)
    
    with gr.Blocks(title="Drone Detection Model Comparison") as demo:
        gr.Markdown("# 🚁 Drone Detection Model Comparison")
        gr.Markdown("Compare two YOLO models for drone detection side-by-side")
        
        with gr.Tab("Image Comparison"):
            with gr.Row():
                image_input = gr.Image(label="Upload Image", type="numpy")
            
            with gr.Row():
                image_btn = gr.Button("Run Detection", variant="primary")
            
            with gr.Row():
                output1 = gr.Image(label=f"Model 1: {comparator.model1_name}")
                output2 = gr.Image(label=f"Model 2: {comparator.model2_name}")
            
            image_stats = gr.Markdown()
            
            image_btn.click(
                fn=comparator.process_image,
                inputs=[image_input],
                outputs=[output1, output2, image_stats]
            )
        
        with gr.Tab("Video Comparison"):
            gr.Markdown("""
            ### 📹 Synchronized Video Playback
            Videos will play in sync. Use the controls on either video to control both.
            
            **File Size Limit**: Maximum 100MB per video
            
            **Note**: Processed videos are temporary and will be deleted when you close the browser or refresh the page.
            """)
            
            with gr.Row():
                video_input = gr.Video(label="Upload Video (Max 100MB)")
            
            with gr.Row():
                video_btn = gr.Button("Run Detection", variant="primary")
            
            with gr.Row():
                video_output1 = gr.Video(label=f"Model 1: {comparator.model1_name}", autoplay=False, elem_id="video1")
                video_output2 = gr.Video(label=f"Model 2: {comparator.model2_name}", autoplay=False, elem_id="video2")
            
            video_stats = gr.Markdown()
            
            with gr.Row():
                graph_output = gr.Image(label="Detections Per Frame Comparison", type="filepath")
            
            # Add synchronization script
            gr.HTML("""
            <script>
            function setupVideoSync() {
                // Wait for videos to load
                const checkVideos = setInterval(() => {
                    const videos = document.querySelectorAll('video');
                    if (videos.length >= 2) {
                        clearInterval(checkVideos);
                        const video1 = videos[videos.length - 2];
                        const video2 = videos[videos.length - 1];
                        
                        let syncing = false;
                        
                        // Sync from video1 to video2
                        video1.addEventListener('play', () => {
                            if (!syncing) {
                                syncing = true;
                                video2.currentTime = video1.currentTime;
                                video2.play().catch(() => {});
                                syncing = false;
                            }
                        });
                        
                        video1.addEventListener('pause', () => {
                            if (!syncing) {
                                syncing = true;
                                video2.pause();
                                syncing = false;
                            }
                        });
                        
                        video1.addEventListener('seeked', () => {
                            if (!syncing) {
                                syncing = true;
                                video2.currentTime = video1.currentTime;
                                syncing = false;
                            }
                        });
                        
                        // Sync from video2 to video1
                        video2.addEventListener('play', () => {
                            if (!syncing) {
                                syncing = true;
                                video1.currentTime = video2.currentTime;
                                video1.play().catch(() => {});
                                syncing = false;
                            }
                        });
                        
                        video2.addEventListener('pause', () => {
                            if (!syncing) {
                                syncing = true;
                                video1.pause();
                                syncing = false;
                            }
                        });
                        
                        video2.addEventListener('seeked', () => {
                            if (!syncing) {
                                syncing = true;
                                video1.currentTime = video2.currentTime;
                                syncing = false;
                            }
                        });
                        
                        console.log('Video synchronization enabled');
                    }
                }, 500);
                
                // Clear interval after 10 seconds if videos not found
                setTimeout(() => clearInterval(checkVideos), 10000);
            }
            
            // Run on page load and after updates
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', setupVideoSync);
            } else {
                setupVideoSync();
            }
            
            // Re-run when new content is added
            const observer = new MutationObserver(setupVideoSync);
            observer.observe(document.body, { childList: true, subtree: true });
            </script>
            """)
            
            video_btn.click(
                fn=comparator.process_video,
                inputs=[video_input],
                outputs=[video_output1, video_output2, graph_output, video_stats]
            )
        
        gr.Markdown("""
        ### Model Information
        - **Model 1**: Thermal/IR trained model
        - **Model 2**: RGB trained model (or comparison model)
        - **Confidence Threshold**: 0.25
        - **IoU Threshold**: 0.5
        
        ### 💾 Video Storage
        - Processed videos are stored **temporarily** in system temp directory
        - Videos are **automatically deleted** when you close the browser or refresh the page
        - Files are only kept in memory during your current session
        - Download videos if you want to keep them
        
        ### 🎬 Synchronized Playback
        - Both videos play together automatically
        - Control either video to control both
        - Seeking/pausing one video affects the other
        """)
    
    return demo


if __name__ == "__main__":
    # Model paths
    model1_path = "weights/best.pt"
    model2_path = "VisionSentry-dex-rgb-model-replication/weights/best.pt"
    
    # Create and launch interface
    demo = create_interface(model1_path, model2_path)
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
