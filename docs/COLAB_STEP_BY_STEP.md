# Colab Step-by-Step (Fast Baseline)

This guide prepares this workspace to run in Colab quickly with a stable baseline workflow.

## Goal

- Generate a playable baseline video in Colab with minimal setup friction.
- Save outputs to Google Drive.

## What changed in code

- `TTSModule` supports fast mode with `TTS_FAST_MODE=1`.
- `CursorModule` supports fast mode with `CURSOR_FAST_MODE=1`.
- In fast mode, heavy model loading is skipped to reduce failures and startup time.

## Step 1: Open Colab runtime

- Runtime type: GPU
- Internet: enabled

Check GPU:

```python
!nvidia-smi
```

## Step 2: Mount Google Drive

```python
from google.colab import drive

drive.mount('/content/drive')
```

## Step 3: Upload and extract repository ZIP

```python
from google.colab import files
uploaded = files.upload()  # upload your project zip
```

```python
import glob
import os
import zipfile

zip_path = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall('/content')

repo_candidates = [
    p for p in glob.glob('/content/*')
    if os.path.isdir(p) and os.path.exists(os.path.join(p, 'requirements.txt'))
]
assert repo_candidates, 'No project folder with requirements.txt found.'
repo_dir = repo_candidates[0]
print('Repo:', repo_dir)
%cd $repo_dir
```

## Step 4: Install system packages and fast dependencies

```python
!apt-get -qq update
!apt-get -qq install -y ffmpeg libreoffice poppler-utils
!pip -q install -U pip
!pip -q install -r requirements-colab-fast.txt
```

## Step 5: Enable fast mode

```python
%env TTS_FAST_MODE=1
%env CURSOR_FAST_MODE=1
```

## Step 6: Configure API key safely

```python
import getpass
from pathlib import Path

key = getpass.getpass('GEMINI_API_KEY: ')
Path('config/.env').write_text(f'GEMINI_API_KEY={key}\n', encoding='utf-8')
print('config/.env ready')
```

## Step 7: Verify Gemini connectivity

```python
import yaml
from src.config_schema import AppConfig
from src.gemini_client import GeminiClient

cfg = AppConfig(**yaml.safe_load(open('config/default.yaml', encoding='utf-8')))
client = GeminiClient(cfg.llm)
client.load()
print(client.generate('Reply with OK only.')[:80])
```

## Step 8: Run baseline pipeline

```python
!python -m scripts.T2V_pipeline \
  -r "Self-Attention Mechanism" \
  -p "High schooler, no calculus." \
  -o demo.mp4 \
  -d ./output
```

## Step 9: Copy outputs to Google Drive

```python
DRIVE_OUT = '/content/drive/MyDrive/TeachingMonster/output'
!mkdir -p "$DRIVE_OUT"
!cp -f ./output/demo.mp4 "$DRIVE_OUT/"
!ls -lh "$DRIVE_OUT"
```

## Optional: Keep debug artifacts

```python
DRIVE_LOG = '/content/drive/MyDrive/TeachingMonster/logs'
!mkdir -p "$DRIVE_LOG"
!cp -r ./tmp "$DRIVE_LOG/" || true
!cp -r ./tmp_pipeline "$DRIVE_LOG/" || true
```

## Notes

- Fast mode is optimized for speed and baseline stability.
- For final competition quality, disable fast mode and install full dependencies from `requirements.txt`.
- Colab sessions are not long-running API servers. Use a cloud VM for final submission hosting.
