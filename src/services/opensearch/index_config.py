"""Index definition for segment documents. One document = one segment (a searchable moment of a video)."""

INDEX_VERSION = 1


def versioned_index_name(alias: str, version: int = INDEX_VERSION) -> str:
    """Clients always use the alias; the real index is versioned so it can be rebuilt and swapped with no downtime."""
    return f"{alias}_v{version}"


INDEX_BODY = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,  # single node locally; production sets >= 1
        "index.knn": True,  # can only be set at creation: enables vector fields in Week 4 without a rebuild
        "analysis": {
            "analyzer": {
                # Keeps every word ("is", "the") and its position: required for exact phrase matching of quotes.
                "transcript_exact": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "asciifolding"]},
            }
        },
    },
    "mappings": {
        "dynamic": "strict",  # unknown fields are rejected instead of silently creating a messy mapping
        "properties": {
            "segment_id": {"type": "keyword"},
            "video_id": {"type": "keyword"},
            "kind": {"type": "keyword"},
            "idx": {"type": "integer"},
            "start_sec": {"type": "float"},
            "end_sec": {"type": "float"},
            # Transcript, two ways: exact (quotes) + English-stemmed (loose keyword search: "changing" ~ "change")
            "text": {
                "type": "text",
                "analyzer": "transcript_exact",
                "fields": {"stemmed": {"type": "text", "analyzer": "english"}},
            },
            "words": {"type": "object", "enabled": False},  # stored for exact timings, not searchable
            "frame_key": {"type": "keyword", "index": False},
            "frame_time_sec": {"type": "float", "index": False},
            # Denormalised video fields: filter/rank without joins, render results without a DB round trip
            "video_title": {"type": "text", "analyzer": "english", "fields": {"raw": {"type": "keyword"}}},
            "video_source": {"type": "keyword"},
            "video_source_id": {"type": "keyword"},
            "video_author": {"type": "keyword", "index": False},
            "video_source_url": {"type": "keyword", "index": False},
            "video_s3_key": {"type": "keyword", "index": False},
            "video_duration_sec": {"type": "float"},
            "language": {"type": "keyword"},
            "indexed_at": {"type": "date"},
        },
    },
}
