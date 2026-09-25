from src.services.processing.segmentation import Word, build_shots, build_speech_windows


def test_build_shots_uses_cuts_as_boundaries():
    shots = build_shots([4.0, 7.5], duration=12.0, min_len=1.0, max_len=10.0)
    assert [(s.start, s.end) for s in shots] == [(0.0, 4.0), (4.0, 7.5), (7.5, 12.0)]


def test_build_shots_ignores_flicker_and_short_tail():
    # 4.3 is too close to 4.0; 11.6 would leave a 0.4 s tail
    shots = build_shots([4.0, 4.3, 11.6], duration=12.0, min_len=1.0, max_len=10.0)
    assert [(s.start, s.end) for s in shots] == [(0.0, 4.0), (4.0, 12.0)]


def test_build_shots_splits_long_shots_evenly():
    shots = build_shots([], duration=25.0, min_len=1.0, max_len=10.0)
    assert [(s.start, s.end) for s in shots] == [(0.0, 8.333), (8.333, 16.667), (16.667, 25.0)]
    assert shots[0].mid == (0.0 + 8.333) / 2


def _words(text: str, start: float = 0.0, step: float = 0.5) -> list[Word]:
    return [Word(f" {w}", start + i * step, start + i * step + 0.4) for i, w in enumerate(text.split())]


def test_speech_windows_overlap_and_keep_word_timestamps():
    words = _words(" ".join(f"w{i}" for i in range(80)))  # 40 s of speech, a word every 0.5 s
    windows = build_speech_windows(words, window_sec=15.0, stride_sec=10.0)

    assert windows[0]["start_sec"] == 0.0 and windows[0]["end_sec"] <= 15.0
    assert windows[1]["start_sec"] == 10.0  # stride
    assert windows[0]["end_sec"] > windows[1]["start_sec"]  # overlap
    assert windows[-1]["words"][-1]["word"] == "w79"  # nothing dropped at the end
    assert windows[0]["words"][0] == {"word": "w0", "start": 0.0, "end": 0.4, "prob": None}


def test_quote_spanning_a_window_boundary_is_whole_in_some_window():
    filler = " ".join(f"f{i}" for i in range(27))  # quote starts at 13.5 s, across the 15 s boundary
    words = _words(f"{filler} AI is changing everything and more words follow here to pad")
    windows = build_speech_windows(words, window_sec=15.0, stride_sec=10.0)
    assert any("AI is changing everything" in w["text"] for w in windows)


def test_speech_windows_empty():
    assert build_speech_windows([], 15.0, 10.0) == []
