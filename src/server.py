"""
FastAPI server for video generation API.

This server implements the POST /v1/video/generate endpoint as specified
in docs/api/video-generation.md
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from scripts.T2V_pipeline import VideoGenerationPipeline
from src import AppConfig, GeminiClient


# =============================
# Request/Response Models
# =============================


class VideoGenerateRequest(BaseModel):
    request_id: str
    course_requirement: str
    student_persona: str


class VideoGenerateResponse(BaseModel):
    video_url: str
    subtitle_url: Optional[str] = None
    supplementary_url: Optional[Union[str, list[str]]] = None


REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "1740"))
MAX_VIDEO_BYTES = 3 * 1024 * 1024 * 1024
MAX_AUX_BYTES = 100 * 1024 * 1024
MAX_SUPP_FILES = 5

# Set DRIVE_OUTPUT_FOLDER_ID to upload videos to Google Drive (required for 48-hour URL validity).
DRIVE_OUTPUT_FOLDER_ID = os.getenv("DRIVE_OUTPUT_FOLDER_ID", "")


# =============================
# Google Drive upload
# =============================


def _upload_file_to_drive(local_path: str, filename: str, folder_id: str, mimetype: str = "video/mp4") -> str:
    """
    Upload a file to Google Drive, make it publicly readable,
    and return a direct download URL valid indefinitely.
    Requires Application Default Credentials (set up via google.colab.auth in the notebook).
    """
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.auth import default

    creds, _ = default(scopes=["https://www.googleapis.com/auth/drive"])
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    file_meta = {"name": filename, "parents": [folder_id]}
    media = MediaFileUpload(local_path, mimetype=mimetype, resumable=True)
    uploaded = service.files().create(body=file_meta, media_body=media, fields="id").execute()
    file_id = uploaded["id"]

    service.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
    ).execute()

    return f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"


# =============================
# Helper Functions
# =============================


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_subtitle(
    scripts: list[str],
    word_timings: list[list[tuple[float, float]]],
    output_path: str,
) -> str:
    """
    Generate SRT subtitle file from scripts and word timings.

    Args:
        scripts: List of narration scripts, one per slide
        word_timings: List of word timings per slide, where each timing is (start, end)
        output_path: Path where SRT file should be saved

    Returns:
        Path to the generated SRT file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    subtitle_entries = []
    entry_index = 1
    cumulative_time = 0.0

    for _slide_idx, (script, timings) in enumerate(
        zip(scripts, word_timings, strict=True)
    ):
        words = script.split()

        if not words or not timings:
            continue

        # Group words into phrases for better readability
        # For simplicity, we'll create one subtitle entry per slide
        # You can modify this to group words more intelligently
        slide_start = cumulative_time
        slide_end = cumulative_time

        # Find the end time of the last word in this slide
        if timings:
            slide_end = max(end for _, end in timings) + cumulative_time

        # Create subtitle entry for this slide
        start_time = format_srt_time(slide_start)
        end_time = format_srt_time(slide_end)

        subtitle_entries.append(
            f"{entry_index}\n{start_time} --> {end_time}\n{script}\n\n"
        )

        entry_index += 1
        cumulative_time = slide_end

    # Write SRT file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(subtitle_entries))

    return output_path


def _ensure_size_limit(file_path: str, max_bytes: int, label: str) -> int:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{label} not found: {file_path}")
    size = os.path.getsize(file_path)
    if size > max_bytes:
        raise ValueError(
            f"{label} exceeds limit: {size} bytes > {max_bytes} bytes"
        )
    return size


def _safe_request_dir(base_dir: str, request_id: str) -> str:
    base = Path(base_dir).resolve()
    target = (base / request_id).resolve()
    if base != target and base not in target.parents:
        raise ValueError("Invalid request id path traversal detected")
    return str(target)


def _collect_supplementary_files(request_output_dir: str) -> list[str]:
    files: list[str] = []
    candidate_names = ["presentation.pptx", "slides.pdf"]
    for name in candidate_names:
        p = os.path.join(request_output_dir, name)
        if os.path.exists(p):
            files.append(p)

    # Keep only the first MAX_SUPP_FILES to meet competition constraints.
    return files[:MAX_SUPP_FILES]


# =============================
# FastAPI App
# =============================

app = FastAPI(
    title="Teaching Video Generation API",
    version="1.0.0",
    description=(
        "API for generating teaching videos based on course requirements and student personas"
    ),
)

# Global pipeline instance (loaded once at startup)
pipeline: Optional[VideoGenerationPipeline] = None
app_config: Optional[AppConfig] = None


@app.on_event("startup")
async def startup_event():
    """Load pipeline and models on server startup."""
    global pipeline, app_config

    # Load config
    config_path = os.getenv("CONFIG_PATH", "config/default.yaml")
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        app_config = AppConfig(**data)

    # Initialize client and pipeline
    client = GeminiClient(app_config.llm)
    pipeline = VideoGenerationPipeline(
        llm_client=client,
        app_config=app_config,
        output_video_name="final_video.mp4",  # Will be overridden per request
    )

    # Load all models (this may take time)
    print("Loading pipeline models...")
    pipeline.load()
    print("Pipeline loaded successfully!")


@app.post("/v1/video/generate", response_model=VideoGenerateResponse)
async def generate_video(request: VideoGenerateRequest, ori_req: Request):
    """
    Generate a teaching video based on course requirements and student persona.

    This endpoint:
    1. Generates video outlines
    2. Creates slides
    3. Generates TTS audio
    4. Creates cursor trajectories
    5. Renders final MP4 video
    6. Generates SRT subtitles
    """
    if ori_req.headers.get("X-Dry-Run") == "true":
        print("[Log] Dry Run / Connection Test received.")
        return {
            "video_url": "https://example.com/test.mp4",
            "subtitle_url": "https://example.com/test.srt",
            "supplementary_url": None
        }
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        # Create output directory for this request
        output_config = app_config.output if app_config else None
        output_base = (
            output_config.final_video_dir if output_config else None
        ) or "./output"

        request_output_dir = _safe_request_dir(output_base, request.request_id)
        os.makedirs(request_output_dir, exist_ok=True)

        # Redirect this request's video output to a per-request directory.
        # Models stay loaded in the global pipeline — do NOT call .load() again.
        video_name = f"{request.request_id}.mp4"
        pipeline.video_output_path = os.path.join(request_output_dir, video_name)

        # Run pipeline (CPU/IO intensive — execute in a thread pool)
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                pipeline.run,
                request.course_requirement,  # requirement_prompt
                request.student_persona,  # persona_prompt
            ),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        video_path = result["final_video_path"]
        _ensure_size_limit(video_path, MAX_VIDEO_BYTES, "video")

        # Generate subtitle file
        subtitle_path = os.path.join(request_output_dir, f"{request.request_id}.srt")
        generate_srt_subtitle(
            scripts=result["scripts"],
            word_timings=result["word_timings"],
            output_path=subtitle_path,
        )

        slides_folder = result.get("slides_folder")
        if slides_folder:
            ppt_src = os.path.join(slides_folder, "presentation.pptx")
            ppt_dst = os.path.join(request_output_dir, "presentation.pptx")
            if os.path.exists(ppt_src) and not os.path.exists(ppt_dst):
                shutil.copy2(ppt_src, ppt_dst)

        subtitle_url: Optional[str] = None
        subtitle_size = 0
        if os.path.exists(subtitle_path):
            subtitle_size = os.path.getsize(subtitle_path)

        supplementary_files = _collect_supplementary_files(request_output_dir)
        supplementary_size = sum(os.path.getsize(p) for p in supplementary_files)

        # Enforce competition limit: subtitle + supplementary <= 100MB.
        # If over limit, degrade gracefully by dropping supplementary first.
        if subtitle_size + supplementary_size > MAX_AUX_BYTES:
            supplementary_files = []
            supplementary_size = 0

        if subtitle_size + supplementary_size > MAX_AUX_BYTES:
            # Subtitle alone is too large, drop subtitle as it is optional.
            try:
                os.remove(subtitle_path)
            except OSError:
                pass
            subtitle_size = 0

        # Build URLs.
        # If DRIVE_OUTPUT_FOLDER_ID is set: upload video to Google Drive so the
        # link stays valid for 48+ hours (required by competition rules).
        # Otherwise fall back to ngrok/localhost (valid only while server runs).
        base_url = os.getenv("BASE_URL", "http://localhost:8000")

        if DRIVE_OUTPUT_FOLDER_ID:
            video_url = _upload_file_to_drive(
                video_path,
                f"{request.request_id}.mp4",
                DRIVE_OUTPUT_FOLDER_ID,
            )
        else:
            video_url = f"{base_url}/v1/files/{request.request_id}/video"

        if subtitle_size > 0 and os.path.exists(subtitle_path):
            if DRIVE_OUTPUT_FOLDER_ID:
                subtitle_url = _upload_file_to_drive(
                    subtitle_path,
                    f"{request.request_id}.srt",
                    DRIVE_OUTPUT_FOLDER_ID,
                    mimetype="text/plain",
                )
            else:
                subtitle_url = f"{base_url}/v1/files/{request.request_id}/subtitle"

        supplementary_url: Optional[Union[str, list[str]]] = None
        if supplementary_files:
            if DRIVE_OUTPUT_FOLDER_ID:
                supp_urls = [
                    _upload_file_to_drive(
                        p,
                        Path(p).name,
                        DRIVE_OUTPUT_FOLDER_ID,
                        mimetype="application/octet-stream",
                    )
                    for p in supplementary_files
                ]
            else:
                supp_urls = [
                    f"{base_url}/v1/files/{request.request_id}/supplementary/{Path(p).name}"
                    for p in supplementary_files
                ]
            supplementary_url = supp_urls[0] if len(supp_urls) == 1 else supp_urls

        return VideoGenerateResponse(
            video_url=video_url,
            subtitle_url=subtitle_url,
            supplementary_url=supplementary_url,
        )

    except asyncio.TimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Generation timed out after {REQUEST_TIMEOUT_SECONDS} seconds. "
                "Competition limit is 30 minutes."
            ),
        ) from e

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Video generation failed: {str(e)}"
        ) from e


@app.get("/v1/files/{request_id}/video")
async def get_video(request_id: str):
    """Serve the generated video file."""
    output_config = app_config.output if app_config else None
    output_base = (
        output_config.final_video_dir if output_config else None
    ) or "./output"

    video_path = os.path.join(output_base, request_id, f"{request_id}.mp4")

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{request_id}.mp4",
    )


@app.get("/v1/files/{request_id}/subtitle")
async def get_subtitle(request_id: str):
    """Serve the generated subtitle file."""
    output_config = app_config.output if app_config else None
    output_base = (
        output_config.final_video_dir if output_config else None
    ) or "./output"

    subtitle_path = os.path.join(output_base, request_id, f"{request_id}.srt")

    if not os.path.exists(subtitle_path):
        raise HTTPException(status_code=404, detail="Subtitle not found")

    return FileResponse(
        subtitle_path,
        media_type="text/srt",
        filename=f"{request_id}.srt",
    )


@app.get("/v1/files/{request_id}/supplementary/{filename}")
async def get_supplementary(request_id: str, filename: str):
    """Serve supplementary files (e.g., PPT/PDF)."""
    output_config = app_config.output if app_config else None
    output_base = (
        output_config.final_video_dir if output_config else None
    ) or "./output"

    req_dir = Path(_safe_request_dir(output_base, request_id)).resolve()
    file_path = (req_dir / filename).resolve()
    if req_dir != file_path.parent:
        raise HTTPException(status_code=403, detail="Invalid supplementary path")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Supplementary file not found")

    allowed_suffixes = {".pptx", ".pdf"}
    if file_path.suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=403, detail="File type not allowed")

    media_type = "application/octet-stream"
    suffix = file_path.suffix.lower()
    if suffix == ".pptx":
        media_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    elif suffix == ".pdf":
        media_type = "application/pdf"

    return FileResponse(str(file_path), media_type=media_type, filename=filename)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
