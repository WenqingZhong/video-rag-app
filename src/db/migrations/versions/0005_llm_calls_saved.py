"""Week 6: record cache hits in llm_calls (tokens and cost the cache saved).

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("llm_calls", sa.Column("saved_tokens", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("llm_calls", sa.Column("saved_cost_usd", sa.Numeric(14, 8), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("llm_calls", "saved_cost_usd")
    op.drop_column("llm_calls", "saved_tokens")
