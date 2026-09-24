from abc import ABC, abstractmethod
from contextlib import AbstractContextManager as ContextManager
from typing import Any

from sqlalchemy.orm import Session


class BaseDatabase(ABC):
    """Base class for database operations."""

    @abstractmethod
    def startup(self) -> bool:
        """Initialize the database connection and return whether it is ready."""

    @abstractmethod
    def teardown(self) -> None:
        """Close the database connection."""

    @abstractmethod
    def get_session(self) -> ContextManager[Session]:
        """Get a database session."""

    @abstractmethod
    def healthcheck(self) -> dict[str, Any]:
        """Return the current database state for API health checks."""


class BaseRepository(ABC):
    """Base repository pattern for data access."""

    def __init__(self, session: Session):
        self.session = session

    @abstractmethod
    def create(self, data: dict[str, Any]) -> Any:
        """Create a new record."""

    @abstractmethod
    def get_by_id(self, record_id: Any) -> Any | None:
        """Get a record by ID."""

    @abstractmethod
    def update(self, record_id: Any, data: dict[str, Any]) -> Any | None:
        """Update a record by ID."""

    @abstractmethod
    def delete(self, record_id: Any) -> bool:
        """Delete a record by ID."""

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> list[Any]:
        """List records with pagination."""
