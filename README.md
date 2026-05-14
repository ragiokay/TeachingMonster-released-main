# Teaching Monster: Baseline & Starter Kit!

## Prerequisite to Run the Baseline
* Gemini API: visit [the website](https://aistudio.google.com/welcome) to get your API Key, and in `config/.env`, insert:
   ```
   GEMINI_API_KEY={YOUR_OWN_API_KEY}
   ```
* Server: An GPU w/ 20GB VRAM is recommended.

## Environment Setup

* This project uses **Python 3.10**, and needs libreoffice & poppler to be installed. The installation subjects to the OS.

* Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the pipeline locally!
```bash
python -m scripts.T2V_pipeline \
  -r "REQUIREMENT_PROMPT" \
  -p "PERSONA_PROMPT" \
  [-c config/default.yaml] \
  [-o final_video.mp4] \
  [-d ./output]
```

## Run in staged mode (checkpoint / resume)

If you want to avoid restarting from stage 1 after failures:

```bash
python -m scripts.T2V_staged \
   -r "REQUIREMENT_PROMPT" \
   -p "PERSONA_PROMPT" \
   --end-stage 2

python -m scripts.T2V_staged \
   -r "REQUIREMENT_PROMPT" \
   -p "PERSONA_PROMPT" \
   --start-stage 3
```

This writes checkpoints in `./tmp/staged` by default.

## Reliability knobs for Gemini

You can tune retry/fallback behavior with environment variables:

- `GEMINI_FALLBACK_MODELS` (default: `gemini-2.5-flash,gemini-1.5-flash`)
- `GEMINI_MAX_ATTEMPTS` (default: `3`)
- `GEMINI_RETRY_BASE_DELAY` (default: `2.0`)

## Run on a cloud GPU machine (recommended)

If you want to avoid local disk usage, run this service on a cloud VM.

Quick start:

```bash
cp config/.env.example config/.env
# Set GEMINI_API_KEY in config/.env

docker build -t teaching-monster:gpu .
docker run --rm --gpus all -p 8000:8000 \
   -v $(pwd)/config:/app/config \
   -v $(pwd)/output:/app/output \
   -v $(pwd)/tmp:/app/tmp \
   -v $(pwd)/tmp_pipeline:/app/tmp_pipeline \
   teaching-monster:gpu
```

Detailed guide:

- `docs/CLOUD_DEPLOY.md`

## Run in Colab (fast baseline)

If you need to get a baseline result quickly in Colab, follow:

- `docs/COLAB_STEP_BY_STEP.md`

This flow uses fast mode flags (`TTS_FAST_MODE=1`, `CURSOR_FAST_MODE=1`) to reduce heavy model setup time and improve Colab stability.