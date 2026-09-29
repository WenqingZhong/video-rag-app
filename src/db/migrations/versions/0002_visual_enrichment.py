"""Visual enrichment: caption + CLIP image embedding on visual segments.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("caption", sa.Text()))
    # Stored in Postgres (source of truth) so the search index can be rebuilt without re-running the models.
    op.add_column("segments", sa.Column("image_embedding", sa.JSON()))
    op.add_column("segments", sa.Column("embedding_model", sa.String(64)))
    op.add_column("segments", sa.Column("caption_model", sa.String(64)))


def downgrade() -> None:
    op.drop_column("segments", "caption_model")
    op.drop_column("segments", "embedding_model")
    op.drop_column("segments", "image_embedding")
    op.drop_column("segments", "caption")
