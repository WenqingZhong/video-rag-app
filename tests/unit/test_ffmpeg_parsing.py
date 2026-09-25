import pytest

from src.services.processing.ffmpeg import FFmpegError, parse_probe, parse_scene_times

SHOWINFO = """
[Parsed_showinfo_2 @ 0x1] n:   0 pts:  61440 pts_time:4.8     duration:512 fmt:yuv420p
[Parsed_showinfo_2 @ 0x1] n:   1 pts: 130560 pts_time:10.2    duration:512 fmt:yuv420p
[Parsed_showinfo_2 @ 0x1] n:   2 pts: 130560 pts_time:10.2    duration:512 fmt:yuv420p
"""


def test_parse_scene_times_dedupes_and_sorts():
    assert parse_scene_times(SHOWINFO) == [4.8, 10.2]


def test_parse_probe_reads_video_and_audio_streams():
    data = {
        "format": {"duration": "14.014"},
        "streams": [
            {"codec_type": "video", "width": 1280, "height": 720, "avg_frame_rate": "30000/1001"},
            {"codec_type": "audio"},
        ],
    }
    result = parse_probe(data)
    assert (result.duration_sec, result.width, result.height, result.fps, result.has_audio) == (14.014, 1280, 720, 29.97, True)


def test_parse_probe_rejects_files_without_video():
    with pytest.raises(FFmpegError):
        parse_probe({"format": {"duration": "3"}, "streams": [{"codec_type": "audio"}]})
