from unittest.mock import MagicMock

from src.services.opensearch import OpenSearchService, build_index_body, parse_version
from src.services.processing.visual import VisualEnricher


def service(alias_target="video_segments_v1", fields=("segment_id", "video_id", "text")):
    client = MagicMock()
    client.indices.get_alias.return_value = {alias_target: {}} if alias_target else {}
    client.indices.exists.return_value = False
    client.indices.get_mapping.return_value = {alias_target or "x": {"mappings": {"properties": {f: {} for f in fields}}}}
    client.count.return_value = {"count": 3}
    return OpenSearchService(client, alias="video_segments", index_body=build_index_body(512), version=2), client


def test_version_parsing_and_outdated_detection():
    assert parse_version("video_segments_v12") == 12 and parse_version("weird") is None
    svc, _ = service("video_segments_v1")
    assert svc.is_outdated()
    svc, _ = service("video_segments_v2")
    assert not svc.is_outdated()


def test_migrate_builds_new_index_then_switches_alias_atomically():
    svc, client = service("video_segments_v1")
    filled = []
    result = svc.migrate(lambda index: filled.append(index))

    client.indices.create.assert_called_once()
    assert client.indices.create.call_args.kwargs["index"] == "video_segments_v2"
    assert filled == ["video_segments_v2"]  # filled BEFORE the switch
    actions = client.indices.update_aliases.call_args.kwargs["body"]["actions"]
    assert actions == [
        {"remove": {"index": "video_segments_v1", "alias": "video_segments"}},
        {"add": {"index": "video_segments_v2", "alias": "video_segments"}},
    ]  # one request: searches never see "no index"
    assert result == {"from": "video_segments_v1", "to": "video_segments_v2", "documents": 3}


def test_new_fields_are_dropped_when_writing_to_an_older_index(monkeypatch):
    svc, _ = service("video_segments_v1", fields=("segment_id", "video_id", "text"))
    sent = []
    monkeypatch.setattr("src.services.opensearch.client.helpers.bulk", lambda client, actions, **kw: sent.extend(actions))
    svc.index_video("v1", [{"segment_id": "s1", "video_id": "v1", "text": "hi", "image_embedding": [0.1], "caption": None}])
    assert sent[0]["_source"] == {"segment_id": "s1", "video_id": "v1", "text": "hi"}  # v1 would reject image_embedding


def test_mapping_has_knn_vector_with_configured_dimension():
    field = build_index_body(512)["mappings"]["properties"]["image_embedding"]
    assert field["type"] == "knn_vector" and field["dimension"] == 512 and field["method"]["space_type"] == "cosinesimil"


def test_visual_enricher_adds_vectors_and_captions():
    embedder, captioner = MagicMock(), MagicMock()
    embedder.embed_images.return_value = [[0.1], [0.2]]
    embedder.model = "clip"
    captioner.caption.side_effect = ["a dog", "a cat"]
    captioner.model = "qwen"
    stages = []
    fields = VisualEnricher(embedder, captioner).enrich([b"1", b"2"], on_stage=stages.append)
    assert stages == ["embedding_frames", "captioning"]
    assert fields[1] == {"image_embedding": [0.2], "embedding_model": "clip", "caption": "a cat", "caption_model": "qwen"}
