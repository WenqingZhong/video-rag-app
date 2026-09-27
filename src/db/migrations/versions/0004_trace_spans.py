"""Week 6: trace spans (timed steps of each request and task).

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "trace_spans",
        sa.Column("span_id", sa.String(16), primary_key=True),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("parent_id", sa.String(16)),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("service", sa.String(16), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("attributes", sa.JSON(), nullable=False),
    )
    op.create_index("ix_trace_spans_trace_id", "trace_spans", ["trace_id"])
    op.create_index("ix_trace_spans_name_started_at", "trace_spans", ["name", "started_at"])
    op.create_index("ix_trace_spans_started_at", "trace_spans", ["started_at"])


def downgrade() -> None:
    op.drop_table("trace_spans")
