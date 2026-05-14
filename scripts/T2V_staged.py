"""
Staged Text-to-Video runner with checkpoint/resume support.

Stages:
1. outline
2. wrapper
3. slides
4. tts
5. cursor
6. render

Usage example:
python -m scripts.T2V_staged -r "Self-Attention Mechanism" -p "High schooler" --end-stage 2
python -m scripts.T2V_staged -r "Self-Attention Mechanism" -p "High schooler" --start-stage 3
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, List

import numpy as np
import yaml
from PIL import Image
from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips

from src import AppConfig, GeminiClient, CursorModule, SlidesModule_PPT, T2VOutlineModule, TTSModule, Wrapper_PPT


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_slide_images(slides_folder: str, slide_count: int) -> tuple[list[Image.Image], list[str]]:
    slide_images: list[Image.Image] = []
    slide_image_paths: list[str] = []
    for idx in range(1, slide_count + 1):
        img_path = os.path.join(slides_folder, f"{idx}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing slide image: {img_path}")
        slide_image_paths.append(img_path)
        slide_images.append(Image.open(img_path))
    return slide_images, slide_image_paths


def _render_video(
    output_video_path: str,
    slide_image_paths: List[str],
    audio_paths: List[str],
    cursor_script: List[List[str]],
    cursor_data: List[List[List[int]]],
    word_timings: List[List[List[float]]],
    fps: int = 15,
) -> None:
    video_clips = []

    for slide_idx in range(len(slide_image_paths)):
        img_path = slide_image_paths[slide_idx]
        audio_path = audio_paths[slide_idx]

        audio_clip = AudioFileClip(audio_path).set_fps(44100)
        slide_duration = audio_clip.duration

        base_clip = ImageClip(img_path).set_duration(slide_duration)

        cursor_img = np.zeros((20, 20, 3), dtype=np.uint8)
        cursor_img[:, :] = (255, 0, 0)
        cursor_clip = ImageClip(cursor_img).set_duration(slide_duration)

        cursor_move_time = 0.4
        cursor_move_frame = 10
        prev_x, prev_y = None, None
        word_cnt = 0
        grouped_script = cursor_script[slide_idx]
        grouped_point = cursor_data[slide_idx]
        timestamps = word_timings[slide_idx]

        positions = []

        for script, (x, y) in zip(grouped_script, grouped_point):
            start_time = 0.0 if word_cnt == 0 else timestamps[word_cnt][0]

            if prev_x is not None and prev_y is not None:
                vec_x = (x - prev_x) / cursor_move_frame
                vec_y = (y - prev_y) / cursor_move_frame

                for frame_i in range(cursor_move_frame):
                    t = start_time + frame_i * (cursor_move_time / cursor_move_frame)
                    pos_x = int(prev_x + frame_i * vec_x)
                    pos_y = int(prev_y + frame_i * vec_y)
                    positions.append((t, (pos_x, pos_y)))

                start_time += cursor_move_time

            word_cnt += len(script.split())
            prev_x, prev_y = x, y
            end_t = timestamps[word_cnt - 1][1] if word_cnt - 1 < len(timestamps) else slide_duration
            positions.append((end_t, (x, y)))

        def make_position_fn(pos_list):
            def pos_fn(t):
                for i in range(len(pos_list) - 1):
                    if pos_list[i][0] <= t < pos_list[i + 1][0]:
                        return pos_list[i][1]
                return pos_list[-1][1]

            return pos_fn

        cursor_clip = cursor_clip.set_position(make_position_fn(positions))
        video_clips.append(CompositeVideoClip([base_clip, cursor_clip]).set_audio(audio_clip))

    final_video = concatenate_videoclips(video_clips, method="compose")
    final_video.write_videofile(
        output_video_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
    )

    final_video.close()
    for clip in video_clips:
        clip.close()
        if hasattr(clip, "audio") and clip.audio:
            clip.audio.close()


def main():
    parser = argparse.ArgumentParser(description="Run staged T2V pipeline with checkpoint/resume")
    parser.add_argument("-r", "--requirement-prompt", type=str, required=True)
    parser.add_argument("-p", "--persona-prompt", type=str, required=True)
    parser.add_argument("-c", "--config", type=str, default="config/default.yaml")
    parser.add_argument("-o", "--output-video-name", type=str, default="final_video.mp4")
    parser.add_argument("-d", "--final-video-dir", type=str, default="./output")
    parser.add_argument("--work-dir", type=str, default="./tmp/staged")
    parser.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--end-stage", type=int, default=6, choices=[1, 2, 3, 4, 5, 6])

    args = parser.parse_args()

    if args.start_stage > args.end_stage:
        raise ValueError("start-stage must be <= end-stage")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    config = AppConfig(**data)

    client = GeminiClient(config.llm)

    output_config = config.output
    tmp_dir = (output_config.tmp_dir if output_config else None) or "./tmp_pipeline"
    os.makedirs(tmp_dir, exist_ok=True)
    final_dir = args.final_video_dir or (output_config.final_video_dir if output_config else None) or "./output"
    os.makedirs(final_dir, exist_ok=True)

    slides_output_root = os.path.join(tmp_dir, "slides")
    tts_output_root = os.path.join(tmp_dir, "tts")
    cursor_output_root = os.path.join(tmp_dir, "cursor")

    outline_module = T2VOutlineModule(client)
    wrapper = Wrapper_PPT(client)
    slides_module = SlidesModule_PPT(client, config=config.ppt, output_root=slides_output_root)
    tts_module = TTSModule(output_root=tts_output_root)
    cursor_module = CursorModule(
        output_root=cursor_output_root,
        final_root=final_dir,
        final_video_name=args.output_video_name,
    )

    outline_module.load()
    wrapper.load()
    slides_module.load()
    tts_module.load()
    cursor_module.load()

    outlines_path = work_dir / "outlines.json"
    slides_struct_path = work_dir / "slides_struct.json"
    scripts_path = work_dir / "scripts.json"
    word_timings_path = work_dir / "word_timings.json"
    cursor_script_path = work_dir / "cursor_script.json"
    cursor_data_path = work_dir / "cursor_data.json"
    state_path = work_dir / "state.json"

    outlines = None
    slides_struct = None
    scripts = None
    word_timings = None
    cursor_script = None
    cursor_data = None

    if args.start_stage <= 1 <= args.end_stage:
        outlines = outline_module.run(
            requirement_prompt=args.requirement_prompt,
            persona_prompt=args.persona_prompt,
        )
        _save_json(outlines_path, outlines)
    else:
        outlines = _load_json(outlines_path)

    if args.start_stage <= 2 <= args.end_stage:
        slides_struct, scripts = wrapper.run(outlines)
        _save_json(slides_struct_path, slides_struct)
        _save_json(scripts_path, scripts)
    else:
        slides_struct = _load_json(slides_struct_path)
        scripts = _load_json(scripts_path)

    if args.start_stage <= 3 <= args.end_stage:
        slides_folder = slides_module.run(slides_struct)
    else:
        slides_folder = slides_output_root

    _save_json(state_path, {"slides_folder": slides_folder, "final_dir": final_dir})

    slide_images, slide_image_paths = _load_slide_images(slides_folder, len(slides_struct))

    if args.start_stage <= 4 <= args.end_stage:
        audio_folder, word_timings = tts_module.run(scripts)
        _save_json(word_timings_path, word_timings)
    else:
        audio_folder = tts_output_root
        word_timings = _load_json(word_timings_path)

    audio_paths = [
        os.path.join(audio_folder, f"{i}.mp3") for i in range(1, len(scripts) + 1)
    ]

    if args.start_stage <= 5 <= args.end_stage:
        cursor_script, cursor_data = cursor_module.run(
            images=slide_images,
            scripts=scripts,
            timestamps=word_timings,
            audio_paths=audio_paths,
        )
        _save_json(cursor_script_path, cursor_script)
        _save_json(cursor_data_path, cursor_data)
    else:
        cursor_script = _load_json(cursor_script_path)
        cursor_data = _load_json(cursor_data_path)

    if args.start_stage <= 6 <= args.end_stage:
        video_output_path = os.path.join(final_dir, args.output_video_name)
        _render_video(
            output_video_path=video_output_path,
            slide_image_paths=slide_image_paths,
            audio_paths=audio_paths,
            cursor_script=cursor_script,
            cursor_data=cursor_data,
            word_timings=word_timings,
            fps=15,
        )
        print("Final video:", video_output_path)


if __name__ == "__main__":
    main()
