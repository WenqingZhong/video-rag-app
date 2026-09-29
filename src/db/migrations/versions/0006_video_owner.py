"""Video ownership: an upload is visible only to its owner (NULL = the shared library).

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing videos keep NULL: they stay in the shared library.
    op.add_column("videos", sa.Column("owner_id", sa.String(64)))
    op.create_index("ix_videos_owner_id", "videos", ["owner_id"])


def downgrade() -> None:
    op.drop_index("ix_videos_owner_id", table_name="videos")
    op.drop_column("videos", "owner_id")
