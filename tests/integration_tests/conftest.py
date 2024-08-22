import os

import pytest
from langgraph_sdk import get_client
from langgraph_storage.database import connect

from langgraph_api.shared.utils import AsyncConnectionProto


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def client(anyio_backend):
    return get_client(url=os.getenv("LANGGRAPH_ENDPOINT", "http://localhost:9123"))


@pytest.fixture(scope="session")
async def conn(anyio_backend):
    async with connect(__test__=True) as conn:
        yield conn


@pytest.fixture(scope="function", autouse=True)
async def clear_test_db(anyio_backend, conn: AsyncConnectionProto):
    """Truncate all tables before each test."""
    await conn.execute("DELETE FROM thread")
    await conn.execute(
        "DELETE FROM assistant WHERE metadata->>'created_by' is null OR metadata->>'created_by' != 'system'"
    )
