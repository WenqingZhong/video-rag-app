import logging
from collections.abc import Generator
from contextlib import contextmanager

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.db.interfaces.base import BaseDatabase

logger = logging.getLogger(__name__)


class PostgreSQLSettings(BaseSettings):
    """PostgreSQL configuration settings."""

    database_url: str = Field(
        default="postgresql://video_rag:video_rag_password@localhost:5432/video_rag",
        description="PostgreSQL database URL",
    )
    echo_sql: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=20, description="Database connection pool size")
    max_overflow: int = Field(default=0, description="Maximum pool overflow")

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")


Base = declarative_base()


class PostgreSQLDatabase(BaseDatabase):
    """PostgreSQL database implementation."""

    def __init__(self, config: PostgreSQLSettings):
        self.config = config
        self.engine: Engine | None = None
        self.session_factory: sessionmaker | None = None
        self.last_error: str | None = None

    def startup(self) -> bool:
        """Initialize the database connection and return availability."""
        try:
            logger.info(
                "Attempting to connect to PostgreSQL at: %s",
                self.config.database_url.split("@")[1] if "@" in self.config.database_url else "localhost",
            )

            self.engine = create_engine(
                self.config.database_url,
                echo=self.config.echo_sql,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,
            )

            self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

            assert self.engine is not None
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info("Database connection test successful")

            from src.db.migrations.runner import run_migrations  # alembic is only needed here

            run_migrations(self.engine)
            logger.info("Schema is up to date (tables: %s)", ", ".join(sorted(inspect(self.engine).get_table_names())))

            self.last_error = None
            logger.info("PostgreSQL database initialized successfully")
            logger.info("Database: %s", self.engine.url.database)
            return True

        except Exception as exc:  # noqa: BLE001 - any failure means "degraded", never crash startup/health
            self.engine = None
            self.session_factory = None
            self.last_error = str(exc)
            logger.warning("PostgreSQL unavailable during startup; app will continue in degraded mode: %s", exc)
            return False

    def teardown(self) -> None:
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.session_factory = None
            logger.info("PostgreSQL database connections closed")

    def healthcheck(self) -> dict:
        """Report the current database health for app health checks."""
        if self.engine is None or self.session_factory is None:
            return {
                "status": "unavailable",
                "database": self.config.database_url.split("/")[-1].split("?")[0]
                if "/" in self.config.database_url
                else "video_rag",
                "host": self.config.database_url.split("@")[-1].split("/")[0] if "@" in self.config.database_url else "localhost",
                "error": self.last_error or "database_not_initialized",
            }

        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {
                "status": "healthy",
                "database": self.engine.url.database,
                "host": self.engine.url.host,
                "port": self.engine.url.port,
            }
        except Exception as exc:  # noqa: BLE001 - any failure means "degraded", never crash startup/health
            self.last_error = str(exc)
            return {
                "status": "unavailable",
                "database": self.engine.url.database,
                "host": self.engine.url.host,
                "port": self.engine.url.port,
                "error": self.last_error,
            }

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call startup() first.")

        session = self.session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
