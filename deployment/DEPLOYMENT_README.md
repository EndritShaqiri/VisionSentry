# Drone Detection Model Comparison - Deployment Guide

## Quick Start: Hugging Face Spaces (Easiest)

### 1. Create Account
- Go to https://huggingface.co
- Sign up for free

### 2. Create New Space
- Click "Spaces" → "Create new Space"
- Name: `drone-detection-comparison`
- SDK: **Gradio**
- Hardware: CPU basic (free)

### 3. Upload Files
```
app.py (rename from model_comparison_app.py)
requirements_app.txt
weights/best.pt
VisionSentry-dex-rgb-model-replication/weights/best.pt
```

### 4. Done!
Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/drone-detection-comparison`

---

## Why Not Vercel?

**Vercel does NOT support Python applications.** It only supports JavaScript/Node.js frameworks.

For Python ML apps, use:
- ✅ Hugging Face Spaces (recommended)
- ✅ Railway.app
- ✅ Render.com
- ✅ Google Cloud Run
- ❌ Vercel (JavaScript only)

---

## Alternative: Railway.app

1. Go to https://railway.app
2. Sign up with GitHub
3. "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects Dockerfile
6. Click "Deploy"

---

## Alternative: Render.com

1. Go to https://render.com
2. Sign up with GitHub
3. "New +" → "Web Service"
4. Connect repository
5. Environment: Docker
6. Plan: Free
7. Click "Create Web Service"

---

## Local Testing

### Using Docker
```bash
docker build -t drone-detection-app .
docker run -p 7860:7860 drone-detection-app
```

### Using Python
```bash
pip install -r requirements_app.txt
python model_comparison_app.py
```

Open: http://localhost:7860

---

## Configuration

### CPU-Only (Default)
The app is configured to use CPU by default for deployment:
```python
results = model(image, device='cpu')
```

### Model Paths
Edit in `model_comparison_app.py`:
```python
model1_path = "weights/best.pt"
model2_path = "VisionSentry-dex-rgb-model-replication/weights/best.pt"
```

---

## Files Included

- `model_comparison_app.py` - Main application
- `requirements_app.txt` - Python dependencies
- `Dockerfile` - Docker configuration
- `railway.json` - Railway deployment config
- `render.yaml` - Render deployment config
- `deployment_guide.tex` - Full LaTeX guide
- `DEPLOYMENT_README.md` - This file

---

## Troubleshooting

### Out of Memory
- Reduce video size
- Process fewer frames
- Upgrade to paid tier

### Slow Inference
- CPU is slower than GPU
- Consider Hugging Face GPU tier ($0.60/hour)

### Model Not Found
- Check file paths
- Ensure weights are uploaded
- Verify file structure

---

## Cost

### Free Tiers
- Hugging Face: Free CPU (2 vCPU, 16GB RAM)
- Railway: $5 free credit/month
- Render: 750 hours/month free

### Paid (if needed)
- Hugging Face GPU: $0.60/hour
- Railway: Pay per usage
- Render: $7/month starter

---

## Support

- Gradio Docs: https://gradio.app/docs
- Hugging Face: https://huggingface.co/docs/hub/spaces
- YOLO Docs: https://docs.ultralytics.com

---

**Recommended: Start with Hugging Face Spaces (free, easy, ML-optimized)**
