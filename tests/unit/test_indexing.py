from src.models import Segment, Video
from src.services.indexing import segment_documents
from src.services.opensearch import build_index_body


def test_documents_denormalise_video_fields_and_match_the_strict_mapping():
    video = Video(
        id="v1",
        source="pexels",
        source_id="42",
        title="Dogs playing",
        author_name="A",
        source_url="u",
        s3_key="raw/v1/source.mp4",
        duration_sec=14.0,
        language=None,
    )
    segments = [
        Segment(id="s1", video_id="v1", kind="visual", idx=0, start_sec=0.0, end_sec=7.0, frame_key="frames/v1/000003500.jpg", frame_time_sec=3.5),
        Segment(id="s2", video_id="v1", kind="speech", idx=0, start_sec=1.0, end_sec=4.0, text="hello there", words=[{"word": "hello", "start": 1.0, "end": 1.4}]),
    ]  # fmt: skip
    docs = segment_documents(video, segments)

    assert [d["segment_id"] for d in docs] == ["s1", "s2"]
    assert docs[0]["video_title"] == "Dogs playing"  # visual segments without a caption are findable by title
    assert docs[1]["words"][0]["word"] == "hello"
    mapped = set(build_index_body(512)["mappings"]["properties"])
    assert all(set(d) <= mapped for d in docs)  # "dynamic: strict" would reject unknown fields
