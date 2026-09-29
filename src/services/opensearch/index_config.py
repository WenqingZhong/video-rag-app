"""Index definition for segment documents. One document = one segment (a searchable moment of a video).

Bump INDEX_VERSION whenever the mapping changes: POST /admin/reindex then builds the new versioned index
from Postgres and switches the alias to it atomically (blue/green), keeping the old one for rollback.
"""

import copy

INDEX_VERSION = 3  # v1: text only · v2: + caption, CLIP image embedding · v3: + owner_id (private uploads)


def versioned_index_name(alias: str, version: int = INDEX_VERSION) -> str:
    """Clients always use the alias; the real index is versioned so it can be rebuilt and swapped with no downtime."""
    return f"{alias}_v{version}"


def parse_version(index_name: str) -> int | None:
    suffix = index_name.rsplit("_v", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


_BASE_BODY = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,  # single node locally; production sets >= 1
        "index.knn": True,  # enables vector fields; can only be set at creation
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
            # What the keyframe shows, as words (keyword search) and as a vector (semantic search)
            "caption": {"type": "text", "analyzer": "english"},
            "image_embedding": None,  # filled in by build_index_body (dimension comes from settings)
            "embedding_model": {"type": "keyword"},
            "caption_model": {"type": "keyword"},
            # Denormalised video fields: filter/rank without joins, render results without a DB round trip
            "video_title": {"type": "text", "analyzer": "english", "fields": {"raw": {"type": "keyword"}}},
            "video_source": {"type": "keyword"},
            "video_source_id": {"type": "keyword"},
            "owner_id": {"type": "keyword"},  # absent: the shared library; set: visible only to that user
            "video_author": {"type": "keyword", "index": False},
            "video_source_url": {"type": "keyword", "index": False},
            "video_s3_key": {"type": "keyword", "index": False},
            "video_duration_sec": {"type": "float"},
            "language": {"type": "keyword"},
            "indexed_at": {"type": "date"},
        },
    },
}


def build_index_body(embedding_dim: int) -> dict:
    body = copy.deepcopy(_BASE_BODY)
    body["mappings"]["properties"]["image_embedding"] = {
        "type": "knn_vector",
        "dimension": embedding_dim,
        # HNSW graph: approximate nearest-neighbour search in milliseconds instead of comparing every vector.
        # Lucene engine: supports filters (video_id, source) inside the kNN search itself.
        "method": {
            "name": "hnsw",
            "space_type": "cosinesimil",
            "engine": "lucene",
            "parameters": {"m": 16, "ef_construction": 128},
        },
    }
    return body
