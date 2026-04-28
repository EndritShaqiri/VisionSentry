FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_app.txt .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy application files
COPY drone_detection_app.py .
COPY weights/ ./weights/

# Expose port
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Run the application
CMD ["python", "drone_detection_app.py"]
