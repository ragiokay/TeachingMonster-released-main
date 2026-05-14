# Cloud Deployment (GPU VM)

This guide is the fastest path to run the API on a cloud GPU machine and avoid local disk usage.

## 1. Provision a cloud GPU VM

Recommended baseline:
- Ubuntu 22.04
- NVIDIA GPU with >= 20 GB VRAM
- Disk >= 120 GB
- Docker + NVIDIA Container Toolkit installed

## 2. Clone and prepare secrets

```bash
git clone <your-fork-or-repo-url>
cd TeachingMonster-released-main
cp config/.env.example config/.env
# edit config/.env and set GEMINI_API_KEY
```

## 3. Build container image

```bash
docker build -t teaching-monster:gpu .
```

## 4. Run API container

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/tmp:/app/tmp \
  -v $(pwd)/tmp_pipeline:/app/tmp_pipeline \
  teaching-monster:gpu
```

## 5. Health check

```bash
curl http://<SERVER_IP>:8000/health
```

## 6. Submit test request

```bash
curl -X POST "http://<SERVER_IP>:8000/v1/video/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "smoke-001",
    "course_requirement": "Self-Attention Mechanism",
    "student_persona": "High schooler, no calculus."
  }'
```

## 7. Before competition submission

- Ensure the returned `video_url` is directly downloadable.
- Keep URLs valid for at least 48 hours.
- Keep response latency below 30 minutes per request.
- Keep video <= 3 GB.
- Keep subtitle + supplementary total <= 100 MB.
- Keep supplementary file count <= 5.

## Notes

- This repository's released baseline is PPT-first. Missing optional modules are now guarded at import time so cloud startup is not blocked.
- For stable 48-hour links, upload generated assets to object storage (for example S3-compatible storage) and return those URLs.
