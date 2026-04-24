FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV, matplotlib, and torch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy application files and all runtime dependencies
COPY drone_detection_app.py .
COPY weights/ ./weights/
COPY distance_estimation/ ./distance_estimation/
COPY configs/ ./configs/
COPY src/ ./src/

# Expose default port; Render will also inject PORT
EXPOSE 7860

# Gradio defaults (drone_detection_app.py also honors PORT/HOST env vars)
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Run the application
CMD ["python", "drone_detection_app.py"]
