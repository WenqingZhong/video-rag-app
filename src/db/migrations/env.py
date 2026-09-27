from alembic import context
from sqlalchemy import create_engine

import src.models  # noqa: F401 - register tables on Base.metadata (used by --autogenerate)
from src.config import get_settings
from src.db.interfaces.postgresql import Base

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=get_settings().postgres_database_url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = context.config.attributes.get("connection")  # passed in by the app at startup
    if connection is None:
        engine = create_engine(get_settings().postgres_database_url)
        with engine.connect() as connection:
            _run(connection)
        engine.dispose()
    else:
        _run(connection)


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
