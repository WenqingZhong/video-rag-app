"""Week 6: a ledger of model calls (tokens, time, estimated cost).

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_calls",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("operation", sa.String(32), nullable=False),
        sa.Column("outcome", sa.String(16), nullable=False),
        sa.Column("origin", sa.String(16), nullable=False),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("images", sa.Integer(), nullable=False),
        sa.Column("load_sec", sa.Float(), nullable=False),
        sa.Column("prompt_sec", sa.Float(), nullable=False),
        sa.Column("output_sec", sa.Float(), nullable=False),
        sa.Column("total_sec", sa.Float(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(14, 8), nullable=False),
        sa.Column("price_reference", sa.String(64), nullable=False),
        sa.Column("video_id", sa.String(36)),
        sa.Column("request_id", sa.String(64)),
    )
    op.create_index("ix_llm_calls_created_at", "llm_calls", ["created_at"])
    op.create_index("ix_llm_calls_video_id", "llm_calls", ["video_id"])
    op.create_index("ix_llm_calls_request_id", "llm_calls", ["request_id"])


def downgrade() -> None:
    op.drop_table("llm_calls")
