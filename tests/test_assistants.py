"""Tests for assistants CRUD operations using in-memory SQLite."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from rose_server import database
from rose_server.app import create_app

# Create in-memory engine for testing
test_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    echo=False,
    connect_args={
        "check_same_thread": False,
    },
)

# Create test session factory
test_async_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def test_db():
    """Create test database tables."""
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest.fixture
def client(test_db, monkeypatch):
    """Create test client with in-memory database."""
    # Monkey patch the database session factory
    monkeypatch.setattr(database, "async_session_factory", test_async_session_factory)
    monkeypatch.setattr(database, "engine", test_engine)

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


class TestAssistants:
    """Test Assistants API."""

    def test_create_assistant(self, client):
        """Test creating a new assistant."""
        response = client.post(
            "/v1/assistants",
            json={
                "model": "qwen2.5-0.5b",
                "name": "Test Assistant",
                "description": "A test assistant for unit tests",
                "instructions": "You are a helpful assistant.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "assistant"
        assert data["model"] == "qwen2.5-0.5b"
        assert data["name"] == "Test Assistant"
        assert data["description"] == "A test assistant for unit tests"
        assert data["instructions"] == "You are a helpful assistant."
        assert data["id"].startswith("asst_")

    def test_list_assistants(self, client):
        """Test listing assistants."""
        # Create a few assistants
        for i in range(3):
            client.post(
                "/v1/assistants",
                json={
                    "model": "qwen2.5-0.5b",
                    "name": f"Assistant {i}",
                },
            )

        # List assistants
        response = client.get("/v1/assistants")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 3
        assert all(item["object"] == "assistant" for item in data["data"])

    def test_get_assistant(self, client):
        """Test retrieving a specific assistant."""
        # Create an assistant
        create_response = client.post(
            "/v1/assistants",
            json={
                "model": "qwen2.5-0.5b",
                "name": "Test Assistant",
            },
        )
        assistant_id = create_response.json()["id"]

        # Retrieve the assistant
        response = client.get(f"/v1/assistants/{assistant_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == assistant_id
        assert data["name"] == "Test Assistant"

    def test_update_assistant(self, client):
        """Test updating an assistant."""
        # Create an assistant
        create_response = client.post(
            "/v1/assistants",
            json={
                "model": "qwen2.5-0.5b",
                "name": "Original Name",
                "description": "Original description",
            },
        )
        assistant_id = create_response.json()["id"]

        # Update the assistant
        response = client.post(
            f"/v1/assistants/{assistant_id}",
            json={
                "name": "Updated Name",
                "description": "Updated description",
                "instructions": "New instructions",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == assistant_id
        assert data["name"] == "Updated Name"
        assert data["description"] == "Updated description"
        assert data["instructions"] == "New instructions"
        # Model should not change
        assert data["model"] == "qwen2.5-0.5b"

    def test_delete_assistant(self, client):
        """Test deleting an assistant."""
        # Create an assistant
        create_response = client.post(
            "/v1/assistants",
            json={
                "model": "qwen2.5-0.5b",
                "name": "To Be Deleted",
            },
        )
        assistant_id = create_response.json()["id"]

        # Delete the assistant
        response = client.delete(f"/v1/assistants/{assistant_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == assistant_id
        assert data["object"] == "assistant.deleted"
        assert data["deleted"] is True

        # Verify it's deleted
        response = client.get(f"/v1/assistants/{assistant_id}")
        assert response.status_code == 404

    def test_assistant_not_found(self, client):
        """Test retrieving a non-existent assistant."""
        response = client.get("/v1/assistants/asst_nonexistent")
        assert response.status_code == 404

    def test_assistant_with_tools(self, client):
        """Test creating an assistant with tools."""
        response = client.post(
            "/v1/assistants",
            json={
                "model": "qwen2.5-0.5b",
                "name": "Assistant with Tools",
                "tools": [{"type": "file_search"}, {"type": "code_interpreter"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["tools"]) == 2
        # Check tool types are present
        tool_types = [tool["type"] for tool in data["tools"]]
        assert "file_search" in tool_types
        assert "code_interpreter" in tool_types

    def test_assistant_pagination(self, client):
        """Test pagination for listing assistants."""
        # Create 5 assistants
        for i in range(5):
            client.post(
                "/v1/assistants",
                json={
                    "model": "qwen2.5-0.5b",
                    "name": f"Assistant {i}",
                },
            )

        # Test limit
        response = client.get("/v1/assistants?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2

        # Test full list
        response = client.get("/v1/assistants?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 5

        # Test ordering
        response = client.get("/v1/assistants?order=desc")
        data = response.json()
        # Latest created should be first
        assert "Assistant 4" in data["data"][0]["name"]
