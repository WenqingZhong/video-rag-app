from unittest.mock import MagicMock

from src.services.clips import ClipRange, ClipService, best_sentence, clip_key, clip_range

QUOTE = {"is_quote": True, "match_start": 9.48, "match_end": 10.86, "segment_start": 0.0, "segment_end": 14.48,
         "frame_time": None, "video_duration": 17.44, "padding": 0.75, "max_len": 15.0}  # fmt: skip


def test_quote_clip_is_the_matched_words_plus_padding():
    assert clip_range(**QUOTE) == ClipRange(8.73, 11.61)


def test_clip_is_clamped_to_the_video():
    assert clip_range(**{**QUOTE, "match_start": 0.2, "match_end": 17.3}) == ClipRange(0.0, 17.44)


def test_short_visual_segment_is_used_whole():
    visual = {**QUOTE, "is_quote": False, "segment_start": 7.0, "segment_end": 14.0, "frame_time": 10.5, "max_len": 10.0}
    assert clip_range(**visual) == ClipRange(7.0, 14.0)


def test_long_visual_segment_is_centred_on_the_keyframe():
    visual = {**QUOTE, "is_quote": False, "segment_start": 0.0, "segment_end": 40.0, "frame_time": 30.0,
              "video_duration": 40.0, "max_len": 10.0}  # fmt: skip
    assert clip_range(**visual) == ClipRange(25.0, 35.0)


def test_clip_key_is_deterministic_so_repeats_hit_the_cache():
    assert clip_key("v1", ClipRange(8.73, 11.61)) == "clips/v1/000008730-000011610.mp4"


def test_cached_clip_is_served_without_cutting():
    storage, cut = MagicMock(), MagicMock()
    storage.exists.return_value = True
    assert ClipService(storage, cut).get_or_cut("v1", ClipRange(1.0, 2.0)) == ("clips/v1/000001000-000002000.mp4", True)
    cut.assert_not_called()


def test_missing_clip_is_cut():
    storage, cut = MagicMock(), MagicMock(return_value="clips/v1/000001000-000002000.mp4")
    storage.exists.return_value = False
    assert ClipService(storage, cut).get_or_cut("v1", ClipRange(1.0, 2.0)) == ("clips/v1/000001000-000002000.mp4", False)
    cut.assert_called_once_with("v1", 1.0, 2.0)


# ---- a clip is a moment ---------------------------------------------------------------------------------------------


def test_visual_clip_is_a_few_seconds_around_the_keyframe():
    clip = clip_range(is_quote=False, match_start=0, match_end=10, segment_start=0.0, segment_end=10.0, frame_time=4.0,
                      video_duration=20.0, padding=0.75, max_len=8.0, visual_len=5.0)  # fmt: skip
    assert (clip.start_sec, clip.end_sec) == (1.5, 6.5)


def test_visual_clip_near_the_end_of_a_shot_keeps_its_length():
    clip = clip_range(is_quote=False, match_start=0, match_end=10, segment_start=0.0, segment_end=10.0, frame_time=9.5,
                      video_duration=20.0, padding=0.75, max_len=8.0, visual_len=5.0)  # fmt: skip
    assert (clip.start_sec, clip.end_sec) == (5.0, 10.0)


def test_the_sentence_about_the_topic_is_found():
    spoken = "Hello and welcome. Caffeine has a half-life of about 5 hours. The second is light from screens."
    words = [{"word": w, "start": float(i), "end": i + 0.5} for i, w in enumerate(spoken.split())]
    assert best_sentence(words, "the half-life of caffeine")[:2] == (3.0, 10.5)
    assert best_sentence(words, "dinosaurs") is None


def test_a_very_short_sentence_is_joined_with_the_next():
    words = [
        {"word": w, "start": float(i), "end": i + 0.5}
        for i, w in enumerate(["Naps", "help.", "A", "short", "nap", "of", "twenty", "minutes", "works."])
    ]
    assert best_sentence(words, "naps")[:2] == (0.0, 8.5)
