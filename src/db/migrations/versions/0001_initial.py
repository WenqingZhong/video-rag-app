"""Initial schema: videos and segments.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "videos",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("source_id", sa.String(64)),
        sa.Column("title", sa.String(512)),
        sa.Column("source_url", sa.Text()),
        sa.Column("source_file_url", sa.Text()),
        sa.Column("author_name", sa.String(256)),
        sa.Column("author_url", sa.Text()),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("original_filename", sa.String(512)),
        sa.Column("content_type", sa.String(128)),
        sa.Column("size_bytes", sa.BigInteger()),
        sa.Column("s3_key", sa.Text()),
        sa.Column("duration_sec", sa.Float()),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("fps", sa.Float()),
        sa.Column("has_audio", sa.Boolean()),
        sa.Column("language", sa.String(16)),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("stage", sa.String(32)),
        sa.Column("error", sa.Text()),
        sa.Column("job_id", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("source", "source_id", name="uq_videos_source_source_id"),
    )
    op.create_index("ix_videos_status", "videos", ["status"])
    op.create_index("ix_videos_created_at", "videos", ["created_at"])

    op.create_table(
        "segments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("video_id", sa.String(36), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("text", sa.Text()),
        sa.Column("words", sa.JSON()),
        sa.Column("frame_key", sa.Text()),
        sa.Column("frame_time_sec", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("video_id", "kind", "idx", name="uq_segments_video_kind_idx"),
    )
    op.create_index("ix_segments_video_id_kind", "segments", ["video_id", "kind"])


def downgrade() -> None:
    op.drop_table("segments")
    op.drop_table("videos")
