"""
Client script for the Video Generation API.

This script sends requests to the video generation server and handles responses.
"""

import argparse
import sys
from typing import Optional

import requests


def generate_video(
    base_url: str,
    request_id: str,
    course_requirement: str,
    student_persona: str,
    timeout: Optional[int] = None,
) -> dict:
    """
    Send a video generation request to the server.

    Args:
        base_url: Base URL of the API server (e.g., "http://localhost:8000")
        request_id: Unique ID for tracking this request
        course_requirement: The core educational concept / course requirement
        student_persona: Description of the student's prior knowledge
        timeout: Request timeout in seconds (None for no timeout)

    Returns:
        Response JSON as a dictionary
    """
    url = f"{base_url}/v1/video/generate"

    payload = {
        "request_id": request_id,
        "course_requirement": course_requirement,
        "student_persona": student_persona,
    }

    print(f"\nSending request to {url}")
    print(f"Request ID: {request_id}")
    print(f"Course requirement: {course_requirement}")
    print(f"Student Persona: {student_persona}")
    print("\nGenerating video (this may take several minutes)...")

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()

        result = response.json()
        print("\n=== Video Generation Complete ===")
        print(f"Video URL: {result['video_url']}")
        print(f"Subtitle URL: {result['subtitle_url']}")
        if result.get("supplementary_url"):
            print(f"Supplementary URL: {result['supplementary_url']}")

        return result

    except requests.exceptions.Timeout:
        print(f"\nError: Request timed out after {timeout} seconds")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\nError: HTTP {e.response.status_code}")
        try:
            error_detail = e.response.json()
            print(f"Detail: {error_detail.get('detail', 'Unknown error')}")
        except Exception:
            print(f"Detail: {e.response.text}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"\nError: Failed to connect to server: {e}")
        print(f"Make sure the server is running at {base_url}")
        sys.exit(1)


def download_file(url: str, output_path: str):
    """Download a file from a URL."""
    print(f"Downloading {url} to {output_path}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded successfully to {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Client for Video Generation API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a video
  python scripts/api_client.py \\
    --request-id job-12345 \\
    --course-requirement "Backpropagation in Neural Networks" \\
    --student-persona "Undergraduate CS student, knows Python but new to DL."

  # Generate and download files
  python scripts/api_client.py \\
    --request-id job-12345 \\
    --course-requirement "Machine Learning Basics" \\
    --student-persona "High school student with basic math knowledge" \\
    --download-video \\
    --download-subtitle
        """,
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API server (default: http://localhost:8000)",
    )

    parser.add_argument(
        "--request-id",
        type=str,
        required=True,
        help="Unique ID for tracking (e.g., job-12345)",
    )

    parser.add_argument(
        "--course-requirement",
        type=str,
        required=True,
        dest="course_requirement",
        help="The core educational concept / course requirement",
    )

    parser.add_argument(
        "--student-persona",
        type=str,
        required=True,
        help="Description of the student's prior knowledge",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Request timeout in seconds (default: no timeout)",
    )

    parser.add_argument(
        "--download-video",
        action="store_true",
        help="Download the generated video file",
    )

    parser.add_argument(
        "--download-subtitle",
        action="store_true",
        help="Download the generated subtitle file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./downloads",
        help="Directory to save downloaded files (default: ./downloads)",
    )

    args = parser.parse_args()

    # Generate video
    result = generate_video(
        base_url=args.base_url,
        request_id=args.request_id,
        course_requirement=args.course_requirement,
        student_persona=args.student_persona,
        timeout=args.timeout,
    )

    # Download files if requested
    import os

    if args.download_video or args.download_subtitle:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.download_video:
        video_path = os.path.join(args.output_dir, f"{args.request_id}.mp4")
        download_file(result["video_url"], video_path)

    if args.download_subtitle:
        subtitle_path = os.path.join(args.output_dir, f"{args.request_id}.srt")
        download_file(result["subtitle_url"], subtitle_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
