"""Apply schema migrations at startup (API and worker), safely from many processes at once."""

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent
SCHEMA_LOCK_KEY = 7_311_2024  # app-wide id for pg_advisory_lock: one process migrates, the others wait
BASELINE_REVISION = "0001"  # schema that existed before Alembic was introduced


def _config(connection) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    return config


def run_migrations(engine: Engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": SCHEMA_LOCK_KEY})
        try:
            tables = set(inspect(connection).get_table_names())
            config = _config(connection)
            if "videos" in tables and "alembic_version" not in tables:
                # Database created by create_all before migrations existed: record it as the baseline, don't recreate.
                logger.info("Existing schema without migration history: stamping baseline %s", BASELINE_REVISION)
                command.stamp(config, BASELINE_REVISION)
            command.upgrade(config, "head")
            connection.commit()
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": SCHEMA_LOCK_KEY})
            connection.commit()
