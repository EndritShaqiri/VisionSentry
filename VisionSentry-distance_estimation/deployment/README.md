# Deployment Files

This folder contains all deployment-related files for the Drone Detection Model Comparison app.

## 📄 Files

### Documentation
- **`deployment_guide.pdf`** - Complete deployment guide (10 pages, LaTeX-generated)
- **`DEPLOYMENT_README.md`** - Quick reference deployment instructions
- **`deployment_guide.tex`** - LaTeX source for the PDF guide

### Configuration Files
- **`Dockerfile`** - Docker container configuration
- **`.dockerignore`** - Files to exclude from Docker build
- **`railway.json`** - Railway.app deployment configuration
- **`render.yaml`** - Render.com deployment configuration

## 🚀 Quick Start

1. **Read the PDF guide** (`deployment_guide.pdf`) for complete instructions
2. **Or read** `DEPLOYMENT_README.md` for quick reference
3. **Choose a platform:**
   - Hugging Face Spaces (recommended, easiest)
   - Railway.app
   - Render.com
   - Docker (local or cloud)

## ⚠️ Important

**Vercel does NOT support Python apps.** Use the platforms listed above instead.

## 🔧 Using These Files

### For Hugging Face Spaces
- No config files needed
- Just upload `app.py` and `requirements_app.txt`

### For Railway.app
- Uses `railway.json` automatically
- Detects `Dockerfile`

### For Render.com
- Uses `render.yaml` automatically

### For Docker
- Use `Dockerfile` and `.dockerignore`
- Build: `docker build -t drone-detection-app .`
- Run: `docker run -p 7860:7860 drone-detection-app`

## 📖 Full Documentation

Open `deployment_guide.pdf` for:
- Platform comparisons
- Step-by-step deployment instructions
- Troubleshooting guide
- Performance optimization tips
- Security best practices
- Cost estimates

---

**Start with Hugging Face Spaces** - it's free and optimized for ML apps!
