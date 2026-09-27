"""Thin wrappers around the ffmpeg / ffprobe CLIs (installed in the Docker image)."""

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


@dataclass
class ProbeResult:
    duration_sec: float
    width: int | None
    height: int | None
    fps: float | None
    has_audio: bool


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"{cmd[0]} timed out after {timeout}s") from exc
    if result.returncode != 0:
        raise FFmpegError(f"{cmd[0]} failed: {result.stderr.strip()[-500:]}")
    return result


def _parse_fps(rate: str | None) -> float | None:
    if not rate or rate == "0/0":
        return None
    num, _, den = rate.partition("/")
    return round(float(num) / float(den or 1), 3)


def parse_probe(data: dict) -> ProbeResult:
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError("file has no video stream")
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    if duration <= 0:
        raise FFmpegError("could not determine video duration")
    return ProbeResult(
        duration_sec=duration,
        width=video.get("width"),
        height=video.get("height"),
        fps=_parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def probe(path: Path) -> ProbeResult:
    result = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)], timeout=60)
    return parse_probe(json.loads(result.stdout))


_PTS_TIME = re.compile(r"pts_time:\s*([0-9.]+)")


def parse_scene_times(ffmpeg_stderr: str) -> list[float]:
    """Extract cut timestamps from ffmpeg `showinfo` output (one line per selected frame)."""
    return sorted({round(float(m.group(1)), 3) for m in _PTS_TIME.finditer(ffmpeg_stderr)})


def detect_scene_changes(path: Path, threshold: float, timeout: int = 1800) -> list[float]:
    """Timestamps (s) where the picture changes by more than `threshold`. Downscales first for speed."""
    vf = f"scale=320:-2,select='gt(scene\\,{threshold})',showinfo"
    result = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vf", vf, "-an", "-f", "null", "-"], timeout)
    return parse_scene_times(result.stderr)


def extract_frame(path: Path, time_sec: float, out_path: Path, max_side: int) -> Path:
    """Grab one JPEG at `time_sec`, fitting within max_side × max_side (portrait or landscape).

    Input seeking (-ss before -i) is fast even deep into long videos.
    """
    # Fit the LONG side: scaling only the width made portrait frames 640×1138, doubling captioning cost.
    fit = f"scale=w={max_side}:h={max_side}:force_original_aspect_ratio=decrease:force_divisible_by=2"
    _run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{time_sec:.3f}", "-i", str(path),
         "-frames:v", "1", "-vf", fit, "-q:v", "3", str(out_path)],
        timeout=60,
    )  # fmt: skip
    if not out_path.exists():
        raise FFmpegError(f"no frame extracted at {time_sec:.3f}s")
    return out_path


def extract_audio(path: Path, out_path: Path, timeout: int = 1800) -> Path:
    """16 kHz mono WAV: the format speech-to-text models expect."""
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(out_path),
        ],
        timeout,
    )
    return out_path
