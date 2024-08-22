import asyncio
from base64 import b64encode
from typing import Literal, TypeAlias

import httpx
import pytest
from langgraph_sdk.client import LangGraphClient, Run
from langsmith import Client as LangSmithClient

from langgraph_api.shared.utils import AsyncConnectionProto


class AnyStr(str):
    def __init__(self) -> None:
        super().__init__()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str)

    def __hash__(self) -> int:
        return hash(str(self))


async def poll_run(client: LangGraphClient, thread_id: str, run_id: str) -> Run:
    # Poll until status=success
    run_status = await client.runs.get(thread_id, run_id)
    max_iter = 600
    iter = 0
    while run_status["status"] in ("pending", "running"):
        if iter >= max_iter:
            raise RuntimeError("Max iterations reached")

        await asyncio.sleep(0.1)
        run_status = await client.runs.get(thread_id, run_id)
        iter += 1

    return run_status


# Assistants


async def test_assistant_create_read_update_delete(client: LangGraphClient):
    graph_id = "agent"
    config = {"configurable": {"model_name": "gpt"}}

    create_assistant_response = await client.assistants.create(graph_id, config)
    assert create_assistant_response["graph_id"] == graph_id
    assert create_assistant_response["config"] == config

    get_assistant_response = await client.assistants.get(
        create_assistant_response["assistant_id"]
    )
    assert get_assistant_response["graph_id"] == graph_id
    assert get_assistant_response["config"] == config

    metadata = {"name": "meow"}
    await client.assistants.update(
        create_assistant_response["assistant_id"], graph_id=graph_id, metadata=metadata
    )
    updated_assistant_response = await client.assistants.get(
        create_assistant_response["assistant_id"]
    )
    assert updated_assistant_response["graph_id"] == graph_id
    assert updated_assistant_response["config"] == config
    assert updated_assistant_response["metadata"] == metadata

    await client.assistants.delete(create_assistant_response["assistant_id"])
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.assistants.get(create_assistant_response["assistant_id"])

    assert exc_info.value.response.status_code == 404


async def test_assistant_schemas(client: LangGraphClient):
    graph_id = "other"
    config = {"configurable": {"model": "openai"}}

    create_assistant_response = await client.assistants.create(graph_id, config)
    assert create_assistant_response["graph_id"] == graph_id
    assert create_assistant_response["config"] == config

    get_assistant_response = await client.assistants.get(
        create_assistant_response["assistant_id"]
    )
    assert get_assistant_response["graph_id"] == graph_id
    assert get_assistant_response["config"] == config

    graph = await client.assistants.get_graph(create_assistant_response["assistant_id"])

    assert graph == {
        "nodes": [
            {"id": "__start__", "type": "schema", "data": "__start__"},
            {
                "id": "agent",
                "type": "runnable",
                "data": {
                    "id": ["langgraph", "utils", "RunnableCallable"],
                    "name": "agent",
                },
            },
            {
                "id": "tool",
                "type": "runnable",
                "data": {
                    "id": ["langgraph", "utils", "RunnableCallable"],
                    "name": "tool",
                },
            },
            {"id": "__end__", "type": "schema", "data": "__end__"},
        ],
        "edges": [
            {"source": "__start__", "target": "agent"},
            {"source": "tool", "target": "agent"},
            {"source": "agent", "target": "tool", "conditional": True},
            {"source": "agent", "target": "__end__", "conditional": True},
        ],
    }

    schemas = await client.assistants.get_schemas(
        create_assistant_response["assistant_id"]
    )
    assert schemas["config_schema"] == {
        "title": "Configurable",
        "type": "object",
        "properties": {
            "model": {
                "title": "Model",
                "enum": ["openai", "anthropic"],
                "type": "string",
            }
        },
    }
    assert schemas["state_schema"] == {
        "title": "other_state",
        "type": "object",
        "properties": {
            "messages": {
                "title": "Messages",
                "type": "array",
                "items": {"$ref": "#/definitions/BaseMessage"},
            },
            "other_model": {"$ref": "#/definitions/SomeOtherModel"},
            "sleep": {"title": "Sleep", "type": "integer"},
        },
        "definitions": {
            "BaseMessage": {
                "title": "BaseMessage",
                "description": "Base abstract message class.\n\nMessages are the inputs and outputs of ChatModels.",
                "type": "object",
                "properties": {
                    "content": {
                        "title": "Content",
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {
                                    "anyOf": [{"type": "string"}, {"type": "object"}]
                                },
                            },
                        ],
                    },
                    "additional_kwargs": {
                        "title": "Additional Kwargs",
                        "type": "object",
                    },
                    "response_metadata": {
                        "title": "Response Metadata",
                        "type": "object",
                    },
                    "type": {"title": "Type", "type": "string"},
                    "name": {"title": "Name", "type": "string"},
                    "id": {"title": "Id", "type": "string"},
                },
                "required": ["content", "type"],
            },
            "SomeOtherModel": {
                "title": "SomeOtherModel",
                "type": "object",
                "properties": {"yo": {"title": "Yo", "type": "integer"}},
                "required": ["yo"],
            },
        },
    }
    assert schemas["input_schema"] == {
        "title": "AgentState",
        "type": "object",
        "properties": {
            "messages": {
                "title": "Messages",
                "type": "array",
                "items": {"$ref": "#/definitions/BaseMessage"},
            },
            "other_model": {"$ref": "#/definitions/SomeOtherModel"},
            "sleep": {"title": "Sleep", "type": "integer"},
        },
        "required": ["messages", "other_model"],
        "definitions": {
            "BaseMessage": {
                "title": "BaseMessage",
                "description": "Base abstract message class.\n\nMessages are the inputs and outputs of ChatModels.",
                "type": "object",
                "properties": {
                    "content": {
                        "title": "Content",
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {
                                    "anyOf": [{"type": "string"}, {"type": "object"}]
                                },
                            },
                        ],
                    },
                    "additional_kwargs": {
                        "title": "Additional Kwargs",
                        "type": "object",
                    },
                    "response_metadata": {
                        "title": "Response Metadata",
                        "type": "object",
                    },
                    "type": {"title": "Type", "type": "string"},
                    "name": {"title": "Name", "type": "string"},
                    "id": {"title": "Id", "type": "string"},
                },
                "required": ["content", "type"],
            },
            "SomeOtherModel": {
                "title": "SomeOtherModel",
                "type": "object",
                "properties": {"yo": {"title": "Yo", "type": "integer"}},
                "required": ["yo"],
            },
        },
    }
    assert schemas["output_schema"] == {
        "title": "AgentState",
        "type": "object",
        "properties": {
            "messages": {
                "title": "Messages",
                "type": "array",
                "items": {"$ref": "#/definitions/BaseMessage"},
            },
            "other_model": {"$ref": "#/definitions/SomeOtherModel"},
            "sleep": {"title": "Sleep", "type": "integer"},
        },
        "required": ["messages", "other_model"],
        "definitions": {
            "BaseMessage": {
                "title": "BaseMessage",
                "description": "Base abstract message class.\n\nMessages are the inputs and outputs of ChatModels.",
                "type": "object",
                "properties": {
                    "content": {
                        "title": "Content",
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {
                                    "anyOf": [{"type": "string"}, {"type": "object"}]
                                },
                            },
                        ],
                    },
                    "additional_kwargs": {
                        "title": "Additional Kwargs",
                        "type": "object",
                    },
                    "response_metadata": {
                        "title": "Response Metadata",
                        "type": "object",
                    },
                    "type": {"title": "Type", "type": "string"},
                    "name": {"title": "Name", "type": "string"},
                    "id": {"title": "Id", "type": "string"},
                },
                "required": ["content", "type"],
            },
            "SomeOtherModel": {
                "title": "SomeOtherModel",
                "type": "object",
                "properties": {"yo": {"title": "Yo", "type": "integer"}},
                "required": ["yo"],
            },
        },
    }

    await client.assistants.delete(create_assistant_response["assistant_id"])
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.assistants.get(create_assistant_response["assistant_id"])

    assert exc_info.value.response.status_code == 404


async def test_assistant_schemas_unserializable(client: LangGraphClient):
    graph_id = "agent"
    config = {"configurable": {"model": "openai"}}

    create_assistant_response = await client.assistants.create(graph_id, config)
    assert create_assistant_response["graph_id"] == graph_id
    assert create_assistant_response["config"] == config

    get_assistant_response = await client.assistants.get(
        create_assistant_response["assistant_id"]
    )
    assert get_assistant_response["graph_id"] == graph_id
    assert get_assistant_response["config"] == config

    graph = await client.assistants.get_graph(create_assistant_response["assistant_id"])

    assert graph == {
        "nodes": [
            {"id": "__start__", "type": "schema", "data": "__start__"},
            {
                "id": "agent",
                "type": "runnable",
                "data": {
                    "id": ["langgraph", "utils", "RunnableCallable"],
                    "name": "agent",
                },
            },
            {
                "id": "tool",
                "type": "runnable",
                "data": {
                    "id": ["langgraph", "utils", "RunnableCallable"],
                    "name": "tool",
                },
            },
            {
                "id": "nothing",
                "type": "runnable",
                "data": {
                    "id": ["langgraph", "utils", "RunnableCallable"],
                    "name": "nothing",
                },
            },
            {"id": "__end__", "type": "schema", "data": "__end__"},
        ],
        "edges": [
            {"source": "__start__", "target": "agent"},
            {"source": "tool", "target": "agent"},
            {"source": "agent", "target": "tool", "conditional": True},
            {"source": "agent", "target": "nothing", "conditional": True},
            {"source": "agent", "target": "__end__", "conditional": True},
        ],
    }

    schemas = await client.assistants.get_schemas(
        create_assistant_response["assistant_id"]
    )
    assert schemas["config_schema"] == {}
    assert schemas["state_schema"] == {
        "title": "agent_state",
        "type": "object",
        "properties": {
            "some_bytes": {"title": "Some Bytes", "type": "string", "format": "binary"},
            "some_byte_array": {"title": "Some Byte Array"},  # unserializeable type
            "dict_with_bytes": {
                "title": "Dict With Bytes",
                "type": "object",
                "additionalProperties": {"type": "string", "format": "binary"},
            },
            "messages": {
                "title": "Messages",
                "type": "array",
                "items": {"$ref": "#/definitions/BaseMessage"},
            },
            "sleep": {"title": "Sleep", "type": "integer"},
            "expect_shared_value": {"title": "Expect Shared Value", "type": "boolean"},
        },
        "definitions": {
            "BaseMessage": {
                "title": "BaseMessage",
                "description": "Base abstract message class.\n\nMessages are the inputs and outputs of ChatModels.",
                "type": "object",
                "properties": {
                    "content": {
                        "title": "Content",
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {
                                    "anyOf": [{"type": "string"}, {"type": "object"}]
                                },
                            },
                        ],
                    },
                    "additional_kwargs": {
                        "title": "Additional Kwargs",
                        "type": "object",
                    },
                    "response_metadata": {
                        "title": "Response Metadata",
                        "type": "object",
                    },
                    "type": {"title": "Type", "type": "string"},
                    "name": {"title": "Name", "type": "string"},
                    "id": {"title": "Id", "type": "string"},
                },
                "required": ["content", "type"],
            }
        },
    }
    # currently we don't have a way to serialize input/output schemas
    # with unserializable types
    assert schemas["input_schema"] is None
    assert schemas["output_schema"] is None

    await client.assistants.delete(create_assistant_response["assistant_id"])


async def test_list_assistants(client: LangGraphClient):
    all_assistants = await client.assistants.search()
    assert len(all_assistants) == 2

    graph_id = "agent"
    create_assistant_response = await client.assistants.create(graph_id)
    all_assistants = await client.assistants.search()
    assert len(all_assistants) == 3
    assert (
        create_assistant_response["assistant_id"] == all_assistants[0]["assistant_id"]
    )

    all_assistants = await client.assistants.search(graph_id="agent")
    assert len(all_assistants) == 2
    assert all(a["graph_id"] == graph_id for a in all_assistants)

    all_assistants = await client.assistants.search(metadata={"created_by": "system"})
    assert len(all_assistants) == 2
    assert all(
        create_assistant_response["assistant_id"] != a["assistant_id"]
        for a in all_assistants
    )


# Threads


async def test_thread_create_read_update_delete(client: LangGraphClient):
    metadata = {"name": "test_thread"}

    create_thread_response = await client.threads.create(metadata=metadata)
    assert create_thread_response["metadata"] == metadata

    get_thread_response = await client.threads.get(create_thread_response["thread_id"])
    assert get_thread_response["thread_id"] == create_thread_response["thread_id"]
    assert get_thread_response["metadata"] == metadata

    metadata_update = {"modified": True}
    await client.threads.update(
        create_thread_response["thread_id"], metadata=metadata_update
    )
    updated_thread_response = await client.threads.get(
        create_thread_response["thread_id"]
    )
    assert updated_thread_response["metadata"] == {**metadata, **metadata_update}

    another_thread = await client.threads.create(metadata={"name": "another_thread"})
    all_threads = await client.threads.search()
    assert len(all_threads) == 2
    assert all_threads[0]["thread_id"] == another_thread["thread_id"]
    assert all_threads[1]["thread_id"] == create_thread_response["thread_id"]

    filtered_threads = await client.threads.search(metadata={"modified": True})
    assert len(filtered_threads) == 1
    assert filtered_threads[0]["thread_id"] == create_thread_response["thread_id"]

    await client.threads.delete(create_thread_response["thread_id"])
    all_threads = await client.threads.search()
    assert len(all_threads) == 1
    assert all_threads[0]["thread_id"] == another_thread["thread_id"]


async def test_list_threads(client: LangGraphClient):
    all_threads = await client.threads.search()
    assert len(all_threads) == 0

    # test adding a single thread w/o metadata
    create_thread_response = await client.threads.create()
    all_threads = await client.threads.search()
    assert len(all_threads) == 1
    assert create_thread_response["thread_id"] == all_threads[0]["thread_id"]

    # test adding a thread w/ metadata
    metadata = {"name": "test_thread"}
    create_thread_metadata_response = await client.threads.create(metadata=metadata)
    all_threads = await client.threads.search()
    assert len(all_threads) == 2
    assert create_thread_metadata_response["thread_id"] == all_threads[0]["thread_id"]

    # test filtering on metadata
    filtered_threads = await client.threads.search(metadata=metadata)
    assert len(filtered_threads) == 1
    assert create_thread_metadata_response["thread_id"] == all_threads[0]["thread_id"]

    # test pagination
    paginated_threads = await client.threads.search(offset=1, limit=1)
    assert len(paginated_threads) == 1
    assert create_thread_response["thread_id"] == paginated_threads[0]["thread_id"]


async def test_get_thread_history(client: LangGraphClient):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo"}]}

    empty_history = await client.threads.get_history(thread["thread_id"])
    assert len(empty_history) == 0

    await client.runs.wait(thread["thread_id"], assistant["assistant_id"], input=input)
    history = await client.threads.get_history(thread["thread_id"])
    assert len(history) == 5
    assert len(history[0]["values"]["messages"]) == 4
    assert history[0]["next"] == []
    assert history[-1]["next"] == ["__start__"]

    run_metadata = {"run_metadata": "run_metadata"}
    input = {"messages": [{"role": "human", "content": "bar"}]}
    await client.runs.wait(
        thread["thread_id"],
        assistant["assistant_id"],
        input=input,
        metadata=run_metadata,
    )

    full_history = await client.threads.get_history(thread["thread_id"])
    filtered_history = await client.threads.get_history(
        thread["thread_id"], metadata=run_metadata
    )

    assert len(full_history) == 10
    assert len(full_history[-1]["values"]["messages"]) == 0

    assert len(filtered_history) == 5
    assert len(filtered_history[-1]["values"]["messages"]) == 4


async def test_thread_copy(client: LangGraphClient, conn: AsyncConnectionProto):
    assistant_id = "agent"
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo"}]}
    await client.runs.wait(thread["thread_id"], assistant_id, input=input)
    thread_state = await client.threads.get_state(thread["thread_id"])

    copied_thread = await client.threads.copy(thread["thread_id"])
    copied_thread_state = await client.threads.get_state(copied_thread["thread_id"])

    # check copied thread state matches expected output
    expected_thread_metadata = {
        **thread_state["metadata"],
        "thread_id": copied_thread["thread_id"],
    }
    expected_config = {
        **thread_state["config"],
        "configurable": {
            **thread_state["config"]["configurable"],
            "thread_id": copied_thread["thread_id"],
        },
    }
    expected_parent_config = {
        **thread_state["parent_config"],
        "configurable": {
            **thread_state["parent_config"]["configurable"],
            "thread_id": copied_thread["thread_id"],
        },
    }
    expected_thread_state = {
        **thread_state,
        "metadata": expected_thread_metadata,
        "config": expected_config,
        "parent_config": expected_parent_config,
    }
    assert copied_thread_state == expected_thread_state

    # check checkpoints in DB
    existing_checkpoints = await (
        await conn.execute(
            f"SELECT * FROM checkpoints WHERE thread_id = '{thread['thread_id']}'",
        )
    ).fetchall()
    copied_checkpoints = await (
        await conn.execute(
            f"SELECT * FROM checkpoints WHERE thread_id = '{copied_thread['thread_id']}'",
        )
    ).fetchall()

    assert len(existing_checkpoints) == len(copied_checkpoints)
    for existing, copied in zip(existing_checkpoints, copied_checkpoints):
        existing.pop("thread_id")
        existing["metadata"].pop("thread_id")
        copied.pop("thread_id")
        copied["metadata"].pop("thread_id")
        assert existing == copied

    # check checkpoint blobs in DB
    existing_checkpoint_blobs = await (
        await conn.execute(
            f"SELECT * FROM checkpoint_blobs WHERE thread_id = '{thread['thread_id']}' ORDER BY channel, version",
        )
    ).fetchall()
    copied_checkpoint_blobs = await (
        await conn.execute(
            f"SELECT * FROM checkpoint_blobs WHERE thread_id = '{copied_thread['thread_id']}' ORDER BY channel, version",
        )
    ).fetchall()

    assert len(existing_checkpoint_blobs) == len(copied_checkpoint_blobs)
    for existing, copied in zip(existing_checkpoint_blobs, copied_checkpoint_blobs):
        existing.pop("thread_id")
        copied.pop("thread_id")
        assert existing == copied


async def test_thread_copy_runs(client: LangGraphClient):
    assistant_id = "agent"
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo"}]}
    await client.runs.wait(thread["thread_id"], assistant_id, input=input)
    original_thread_state = await client.threads.get_state(thread["thread_id"])

    copied_thread = await client.threads.copy(thread["thread_id"])
    input = {"messages": [{"role": "human", "content": "bar"}]}
    await client.runs.wait(copied_thread["thread_id"], assistant_id, input=input)

    # test that copied thread has original as well as new values
    copied_thread_state = await client.threads.get_state(copied_thread["thread_id"])
    copied_thread_state_messages = [
        m["content"] for m in copied_thread_state["values"]["messages"]
    ]
    assert copied_thread_state_messages == [
        # original messages
        "foo",
        "begin",
        "tool_call__begin",
        "end",
        # new messages
        "bar",
        "begin",
        "tool_call__begin",
        "end",
    ]

    # test that the new run on the copied thread doesn't affect the original one
    assert await client.threads.get_state(thread["thread_id"]) == original_thread_state


async def test_thread_copy_update(client: LangGraphClient):
    assistant_id = "agent"
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    await client.runs.wait(thread["thread_id"], assistant_id, input=input)
    original_thread_state = await client.threads.get_state(thread["thread_id"])

    copied_thread = await client.threads.copy(thread["thread_id"])

    # update state on a copied thread
    updated_message = {"role": "human", "content": "bar", "id": "initial-message"}
    await client.threads.update_state(
        copied_thread["thread_id"], values={"messages": [updated_message]}
    )
    copied_thread_state = await client.threads.get_state(copied_thread["thread_id"])
    assert copied_thread_state["values"]["messages"][0]["content"] == "bar"

    # test that updating the copied thread doesn't affect the original one
    assert await client.threads.get_state(thread["thread_id"]) == original_thread_state


# Runs


async def test_create_background_run(
    client: LangGraphClient, conn: AsyncConnectionProto
):
    thread = await client.threads.create()
    input = {
        "messages": [{"role": "human", "content": "foo"}],
        "other_model": {"yo": 2},
        "sleep": 1,
    }
    run = await client.runs.create(thread["thread_id"], "other", input=input)

    thread = await client.threads.get(thread["thread_id"])
    assert thread["status"] == "busy"

    # test filtering on status
    filtered_threads = await client.threads.search(status="busy")
    assert len(filtered_threads) == 1
    assert filtered_threads[0]["thread_id"] == thread["thread_id"]
    filtered_threads = await client.threads.search(status="idle")
    assert len(filtered_threads) == 0

    # Poll a single run, status=pending
    run_status = await client.runs.get(thread["thread_id"], run["run_id"])
    assert run_status["status"] == "pending"
    run_status = await poll_run(client, thread["thread_id"], run["run_id"])
    assert run_status["status"] == "success"

    thread = await client.threads.get(thread["thread_id"])
    assert thread["status"] == "idle"

    all_runs = await client.runs.list(thread["thread_id"])
    assert len(all_runs) == 1
    assert all_runs[0]["run_id"] == run["run_id"]
    assert all_runs[0]["status"] == "success"

    cur = await conn.execute(
        f"SELECT * FROM checkpoints WHERE run_id = '{run['run_id']}'",
    )
    assert len(await cur.fetchall()) > 1

    cur = await conn.execute(
        "SELECT * FROM checkpoints WHERE run_id is null",
    )
    assert not await cur.fetchall()


async def test_cancel_background_run(client: LangGraphClient):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo"}], "sleep": 1}
    run = await client.runs.create(
        thread["thread_id"], assistant["assistant_id"], input=input
    )

    await client.runs.cancel(thread["thread_id"], run["run_id"])
    waited_run = await poll_run(client, thread["thread_id"], run["run_id"])
    assert waited_run["status"] == "interrupted"


async def test_cancel_background_run_wait_right_away(client: LangGraphClient):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo"}], "sleep": 1}
    run = await client.runs.create(
        thread["thread_id"], assistant["assistant_id"], input=input
    )

    await client.runs.cancel(thread["thread_id"], run["run_id"], wait=True)
    run_status = await client.runs.get(thread["thread_id"], run["run_id"])
    assert run_status["status"] == "interrupted"


async def test_cancel_background_run_wait_after_start(client: LangGraphClient):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo"}], "sleep": 3}
    run = await client.runs.create(
        thread["thread_id"], assistant["assistant_id"], input=input
    )
    await asyncio.sleep(1)
    await client.runs.cancel(thread["thread_id"], run["run_id"], wait=True)
    run_status = await client.runs.get(thread["thread_id"], run["run_id"])
    assert run_status["status"] == "interrupted"


@pytest.mark.parametrize("wait", [True, False])
async def test_cancel_streaming_run(client: LangGraphClient, wait: bool):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {
        "messages": [{"role": "human", "content": "foo", "id": "initial-message"}],
        "sleep": 3,
    }

    run_id = None
    async for event in client.runs.stream(
        thread["thread_id"], assistant["assistant_id"], input=input
    ):
        if event.event == "error":
            assert event.data == {
                "error": "UserInterrupt",
                "message": "User interrupted the run",
            }
        if event.event == "metadata":
            run_id = event.data["run_id"]
            task = asyncio.create_task(
                client.runs.cancel(thread["thread_id"], run_id, wait=wait)
            )

    await task
    run_status = await client.runs.get(thread["thread_id"], run_id)
    assert run_status["status"] == "interrupted"


@pytest.mark.parametrize("sleep", [0, 0.1, 1])
async def test_interrupt_streaming_run(client: LangGraphClient, sleep: float):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {
        "messages": [{"role": "human", "content": "foo", "id": "initial-message"}],
        "sleep": 3,
    }

    cancelled = False
    run_id = None

    async for event in client.runs.stream(
        thread["thread_id"], assistant["assistant_id"], input=input
    ):
        if event.event == "error":
            raise Exception(event.data)
        if event.event == "metadata":
            run_id = event.data["run_id"]
            cancelled = True
            if sleep:
                await asyncio.sleep(sleep)
            break

    assert cancelled
    run_after = await poll_run(client, thread["thread_id"], run_id)
    # TODO find the edge cases where the status is "error"
    assert run_after["status"] in ("error", "interrupted")


async def test_stream_values(client: LangGraphClient, conn: AsyncConnectionProto):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    stream = client.runs.stream(
        thread["thread_id"],
        assistant["assistant_id"],
        input=input,
        stream_mode="values",
    )

    run_id = None
    previous_message_ids = []
    seen_event_types = set()
    async for chunk in stream:
        seen_event_types.add(chunk.event)

        if chunk.event == "metadata":
            run_id = chunk.data["run_id"]

        if chunk.event == "values":
            message_ids = [message["id"] for message in chunk.data["messages"]]
            assert message_ids[:-1] == previous_message_ids
            previous_message_ids = message_ids

    # make sure the last 'end' event is fired
    assert chunk.event == "end"
    assert seen_event_types == {"metadata", "values", "end"}

    assert run_id is not None
    run = await client.runs.get(thread["thread_id"], run_id)
    assert run["status"] == "success"

    cur = await conn.execute(
        "SELECT * FROM checkpoints WHERE run_id is null",
    )
    assert not await cur.fetchall()

    cur = await conn.execute(
        f"SELECT * FROM checkpoints WHERE run_id = '{run['run_id']}'",
    )
    assert len(await cur.fetchall()) > 1


async def test_stream_updates(client: LangGraphClient):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    stream = client.runs.stream(
        thread["thread_id"],
        assistant["assistant_id"],
        input=input,
        stream_mode="updates",
    )

    run_id = None
    seen_event_types = set()
    seen_nodes = []
    async for chunk in stream:
        seen_event_types.add(chunk.event)

        if chunk.event == "metadata":
            run_id = chunk.data["run_id"]

        if chunk.event == "updates":
            node = list(chunk.data.keys())[0]
            seen_nodes.append(node)

    assert chunk.event == "end"
    assert seen_nodes == ["agent", "nothing", "tool", "agent"]
    assert seen_event_types == {"metadata", "updates", "end"}

    assert run_id is not None
    run = await client.runs.get(thread["thread_id"], run_id)
    assert run["status"] == "success"


async def test_stream_messages(client: LangGraphClient):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    stream = client.runs.stream(
        thread["thread_id"],
        assistant["assistant_id"],
        input=input,
        stream_mode="messages",
    )

    run_id = None
    seen_event_types = set()
    message_id_to_content = {}
    last_message = None
    async for chunk in stream:
        seen_event_types.add(chunk.event)

        if chunk.event == "metadata":
            run_id = chunk.data["run_id"]

        if chunk.event == "messages/partial":
            message = chunk.data[0]
            message_id_to_content[message["id"]] = message["content"]

        if chunk.event == "messages/complete":
            message = chunk.data[0]
            assert message["content"] is not None
            if message["type"] == "ai":
                assert message["content"] == message_id_to_content[message["id"]]

            last_message = message

    assert last_message is not None
    assert last_message["content"] == "end"

    assert chunk.event == "end"
    assert seen_event_types == {
        "metadata",
        "messages/metadata",
        "messages/partial",
        "messages/complete",
        "end",
    }

    assert run_id is not None
    run = await client.runs.get(thread["thread_id"], run_id)
    assert run["status"] == "success"


async def test_stream_mixed_modes(client: LangGraphClient):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    stream = client.runs.stream(
        thread["thread_id"],
        assistant["assistant_id"],
        input=input,
        stream_mode=["messages", "values"],
    )

    run_id = None
    seen_event_types = set()
    messages = []
    async for chunk in stream:
        seen_event_types.add(chunk.event)

        if chunk.event == "metadata":
            run_id = chunk.data["run_id"]

        if chunk.event == "values":
            messages = chunk.data["messages"]

    assert len(messages) == 4
    assert messages[-1]["content"] == "end"

    assert chunk.event == "end"
    assert seen_event_types == {
        "metadata",
        "messages/metadata",
        "messages/partial",
        "messages/complete",
        "values",
        "end",
    }

    assert run_id is not None
    run = await client.runs.get(thread["thread_id"], run_id)
    assert run["status"] == "success"


async def test_stream_run_stateless(
    client: LangGraphClient, conn: AsyncConnectionProto
):
    assistant = await client.assistants.create("agent")
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    stream = client.runs.stream(
        None, assistant["assistant_id"], input=input, stream_mode="values"
    )

    run_id = None
    seen_event_types = set()
    messages = []
    async for chunk in stream:
        seen_event_types.add(chunk.event)

        if chunk.event == "metadata":
            run_id = chunk.data["run_id"]

        if chunk.event == "values":
            messages = chunk.data["messages"]

    assert len(messages) == 4
    assert messages[-1]["content"] == "end"

    assert chunk.event == "end"
    assert seen_event_types == {
        "metadata",
        "values",
        "end",
    }

    # make sure there are no threads left after the execution
    threads = await client.threads.search()
    assert len(threads) == 0

    assert run_id is not None
    cur = await conn.execute(f"SELECT * from run WHERE run_id = '{run_id}'")
    assert await cur.fetchone() is None


async def test_shared_value(client: LangGraphClient):
    assistant = await client.assistants.create("agent")
    # first thread, should set shared value
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    result = await client.runs.wait(
        thread["thread_id"], assistant["assistant_id"], input=input
    )
    assert len(result["messages"]) == 4
    assert result["messages"][-1]["content"] == "end"
    # second thread, should see shared value
    thread = await client.threads.create()
    input = {
        "messages": [{"role": "human", "content": "foo", "id": "initial-message"}],
        "expect_shared_value": True,
    }
    result = await client.runs.wait(
        thread["thread_id"], assistant["assistant_id"], input=input
    )
    assert len(result["messages"]) == 4
    assert result["messages"][-1]["content"] == "end"


async def test_wait_for_run(client: LangGraphClient, conn: AsyncConnectionProto):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    result = await client.runs.wait(
        thread["thread_id"], assistant["assistant_id"], input=input
    )
    assert len(result["messages"]) == 4
    assert result["messages"][-1]["content"] == "end"

    cur = await conn.execute(
        "SELECT * FROM checkpoints WHERE run_id is null",
    )
    assert not await cur.fetchall()


async def test_wait_for_run_stateless(
    client: LangGraphClient, conn: AsyncConnectionProto
):
    assistant = await client.assistants.create("agent")
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    result = await client.runs.wait(None, assistant["assistant_id"], input=input)
    assert len(result["messages"]) == 4
    assert result["messages"][-1]["content"] == "end"

    # make sure there are no threads left after the execution
    threads = await client.threads.search()
    assert len(threads) == 0

    cur = await conn.execute("SELECT * from checkpoints")
    checkpoints = await cur.fetchall()
    assert len(checkpoints) == 0


async def test_background_run_stateless(
    client: LangGraphClient, conn: AsyncConnectionProto
):
    assistant = await client.assistants.create("agent")
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    result = await client.runs.create(None, assistant["assistant_id"], input=input)
    run_id = result["run_id"]

    # poll until deleted
    max_iter = 20
    iter = 0
    while True:
        if iter >= max_iter:
            raise RuntimeError("Max iterations reached")

        await asyncio.sleep(0.5)
        try:
            await client.runs.get(result["thread_id"], run_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                break
            raise
        iter += 1

    cur = await conn.execute(
        f"SELECT * from checkpoints WHERE run_id = '{result['run_id']}'"
    )
    checkpoints = await cur.fetchall()
    assert len(checkpoints) == 0


async def test_background_run_batch_stateless(
    client: LangGraphClient, conn: AsyncConnectionProto
):
    requests = [
        {
            "assistant_id": "agent",
            "input": {
                "messages": [
                    {"role": "human", "content": content, "id": "initial-message"}
                ]
            },
        }
        for content in ["foo", "bar", "baz"]
    ]
    results = await client.runs.create_batch(requests)
    assert len(results) == 3

    run_ids = tuple(result["run_id"] for result in results)
    await asyncio.sleep(0.5)
    cur = await conn.execute(f"SELECT * from checkpoints WHERE run_id IN {run_ids}")
    checkpoints = await cur.fetchall()
    assert len(checkpoints) == 0


@pytest.mark.repeat(10)
async def test_join_successful_background_run(client: LangGraphClient):
    thread = await client.threads.create()
    input = {
        "messages": [{"role": "human", "content": "foo", "id": "initial-message"}],
        "sleep": 1,
    }
    run = await client.runs.create(thread["thread_id"], "agent", input=input)
    await client.runs.join(thread["thread_id"], run["run_id"])


@pytest.mark.repeat(10)
async def test_join_failed_background_run(client: LangGraphClient):
    thread = await client.threads.create()
    run = await client.runs.create(thread["thread_id"], "agent")
    # this run fails because it doesn't have any input
    await client.runs.join(thread["thread_id"], run["run_id"])


@pytest.mark.repeat(10)
async def test_join_streaming_run(client: LangGraphClient):
    thread = await client.threads.create()
    input = {
        "messages": [{"role": "human", "content": "foo", "id": "initial-message"}],
        "sleep": 1,  # sleep for 1 second to ensure join() is called before finish
    }

    scheduled = False
    async with asyncio.TaskGroup() as tg:
        async for event in client.runs.stream(
            thread["thread_id"], "agent", input=input
        ):
            if event.event == "error":
                raise Exception(event.data)
            if event.event == "metadata":
                scheduled = True
                tg.create_task(
                    client.runs.join(thread["thread_id"], event.data["run_id"])
                )
    assert scheduled


async def test_human_in_the_loop(client: LangGraphClient):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}

    # (1) interrupt and then continue running, no modification
    # run until the interrupt
    last_message_before_interrupt = None
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant["assistant_id"],
        input=input,
        interrupt_before=["tool"],
    ):
        if chunk.event == "values":
            last_message_before_interrupt = chunk.data["messages"][-1]
        if chunk.event == "error":
            raise Exception(chunk.data)

    assert last_message_before_interrupt is not None
    assert last_message_before_interrupt["content"] == "begin"

    state = await client.threads.get_state(thread["thread_id"])
    assert state["next"] == ["tool", "nothing"]

    thread = await client.threads.get(thread["thread_id"])
    assert thread["status"] == "interrupted"

    # continue after interrupt
    messages = []
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant["assistant_id"],
        input=None,
    ):
        if chunk.event == "values":
            messages = chunk.data["messages"]
        if chunk.event == "error":
            raise Exception(chunk.data)

    assert chunk.event == "end"
    assert len(messages) == 4
    assert messages[2]["content"] == "tool_call__begin"
    assert messages[-1]["content"] == "end"

    thread = await client.threads.get(thread["thread_id"])
    assert thread["status"] == "idle"

    # (2) interrupt, modify the message and then continue running
    thread = await client.threads.create()

    # run until the interrupt
    last_message_before_interrupt = None
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant["assistant_id"],
        input=input,
        interrupt_before=["tool"],
    ):
        if chunk.event == "values":
            last_message_before_interrupt = chunk.data["messages"][-1]
        if chunk.event == "error":
            raise Exception(chunk.data)

    # edit the last message
    updated_message_content = "modified"
    last_message_before_interrupt["content"] = updated_message_content

    # update state
    await client.threads.update_state(
        thread["thread_id"], {"messages": [last_message_before_interrupt]}
    )
    await client.threads.update(thread["thread_id"], metadata={"modified": True})
    modified_thread = await client.threads.get(thread["thread_id"])
    assert modified_thread["metadata"]["modified"] is True
    state = await client.threads.get_state(thread["thread_id"])
    assert state["values"]["messages"][-1]["content"] == updated_message_content
    assert state["next"] == ["tool", "nothing"]
    assert state["tasks"] == [
        {"id": AnyStr(), "name": "tool", "error": None, "interrupts": []},
        {"id": AnyStr(), "name": "nothing", "error": None, "interrupts": []},
    ]

    # continue after interrupt
    messages = []
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant["assistant_id"],
        input=None,
    ):
        if chunk.event == "values":
            messages = chunk.data["messages"]
        if chunk.event == "error":
            raise Exception(chunk.data)

    assert chunk.event == "end"
    assert len(messages) == 4
    assert messages[2]["content"] == f"tool_call__{updated_message_content}"
    assert messages[-1]["content"] == "end"

    # get the history
    history = await client.threads.get_history(thread["thread_id"])
    assert len(history) == 6
    assert len(history[0]["next"]) == 0
    assert len(history[0]["values"]["messages"]) == 4
    assert history[-1]["next"] == ["__start__"]


async def test_delete_run_checkpoints(
    client: LangGraphClient, conn: AsyncConnectionProto
):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    stream = client.runs.stream(
        thread["thread_id"],
        assistant["assistant_id"],
        input=input,
        stream_mode="values",
    )

    run_id = None
    async for chunk in stream:
        if chunk.event == "error":
            raise Exception(chunk.data)
        if chunk.event == "metadata":
            run_id = chunk.data["run_id"]
            assert run_id is not None

    await client.runs.delete(thread["thread_id"], run_id)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.runs.get(thread["thread_id"], run_id)

    assert exc_info.value.response.status_code == 404

    checkpoint_cur = await conn.execute(
        f"SELECT * FROM checkpoints WHERE run_id = '{run_id}'"
    )
    assert not await checkpoint_cur.fetchall()


async def test_delete_assistant_runs(client: LangGraphClient):
    assistant_1 = await client.assistants.create("agent")
    assistant_2 = await client.assistants.create("agent")
    thread_1 = await client.threads.create()
    thread_2 = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}

    # wait for 2 runs
    await client.runs.wait(
        thread_1["thread_id"], assistant_1["assistant_id"], input=input
    )
    await client.runs.wait(
        thread_2["thread_id"], assistant_2["assistant_id"], input=input
    )

    await client.assistants.delete(assistant_1["assistant_id"])

    assistant_1_runs = await client.runs.list(thread_1["thread_id"])
    assistant_2_runs = await client.runs.list(thread_2["thread_id"])

    assert len(assistant_1_runs) == 0
    assert len(assistant_2_runs) == 1


async def test_delete_thread_runs(client: LangGraphClient, conn: AsyncConnectionProto):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    await client.runs.wait(thread["thread_id"], assistant["assistant_id"], input=input)

    await client.threads.delete(thread["thread_id"])

    cur = await conn.execute(
        f"SELECT * FROM run WHERE thread_id = '{thread['thread_id']}'",
    )
    run_results = await cur.fetchall()
    assert len(run_results) == 0


async def test_create_cron(client: LangGraphClient, conn: AsyncConnectionProto):
    cron_schedule = "* * * * *"
    assistant = await client.assistants.create("agent")
    input = {"messages": [{"role": "human", "content": "foo"}]}
    cron = await client.crons.create(
        assistant["assistant_id"], schedule=cron_schedule, input=input
    )
    cron_id = cron["cron_id"]

    assert cron["schedule"] == cron_schedule
    assert cron["payload"]["assistant_id"] == assistant["assistant_id"]

    cur = await conn.execute(
        f"SELECT * FROM cron WHERE cron_id= '{cron_id}'",
    )
    cron_results = await cur.fetchall()
    assert len(cron_results) == 1

    await client.crons.delete(cron["cron_id"])

    cur = await conn.execute(
        f"SELECT * FROM cron WHERE cron_id = '{cron_id}'",
    )
    cron_results = await cur.fetchall()
    assert len(cron_results) == 0


async def test_create_cron_for_thread(
    client: LangGraphClient, conn: AsyncConnectionProto
):
    cron_schedule = "* * * * *"
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo"}]}
    cron = await client.crons.create_for_thread(
        thread["thread_id"],
        assistant["assistant_id"],
        schedule=cron_schedule,
        input=input,
    )
    cron_id = cron["cron_id"]
    assert cron["schedule"] == cron_schedule
    assert cron["payload"]["assistant_id"] == assistant["assistant_id"]
    assert cron["thread_id"] == thread["thread_id"]

    cur = await conn.execute(
        f"SELECT * FROM cron WHERE cron_id = '{cron_id}'",
    )
    cron_results = await cur.fetchall()
    assert len(cron_results) == 1

    await client.crons.delete(cron["cron_id"])

    cur = await conn.execute(
        f"SELECT * FROM cron WHERE cron_id = '{cron_id}'",
    )
    cron_results = await cur.fetchall()
    assert len(cron_results) == 0


async def test_cron_executions(client: LangGraphClient, conn: AsyncConnectionProto):
    """
    Test cron executions, try with multiple assistants
    """
    cron_schedule = "* * * * * *"  # every second
    assistant = await client.assistants.create("agent")
    assistant2 = await client.assistants.create("agent")
    inputs = {
        assistant["assistant_id"]: {"messages": [{"role": "human", "content": "foo"}]},
        assistant2["assistant_id"]: {"messages": [{"role": "human", "content": "bar"}]},
    }
    threads = {
        assistant["assistant_id"]: await client.threads.create(),
        assistant2["assistant_id"]: await client.threads.create(),
    }
    cron1 = await client.crons.create_for_thread(
        threads[assistant["assistant_id"]]["thread_id"],
        assistant["assistant_id"],
        schedule=cron_schedule,
        input=inputs[assistant["assistant_id"]],
        multitask_strategy="enqueue",
    )
    cron2 = await client.crons.create_for_thread(
        threads[assistant2["assistant_id"]]["thread_id"],
        assistant2["assistant_id"],
        schedule=cron_schedule,
        input=inputs[assistant2["assistant_id"]],
        multitask_strategy="enqueue",
    )
    await asyncio.sleep(2)

    for assistant_id in [assistant["assistant_id"], assistant2["assistant_id"]]:
        cur = await conn.execute(
            f"SELECT * FROM run WHERE assistant_id = '{assistant_id}' and thread_id = '{threads[assistant_id]['thread_id']}' LIMIT 1",
        )
        run_results = await cur.fetchall()
        assert len(run_results) == 1
        assert run_results[0]["kwargs"]["input"] == inputs[assistant_id]

    await client.crons.delete(cron1["cron_id"])
    await client.crons.delete(cron2["cron_id"])


async def test_create_invalid_cron(client: LangGraphClient, conn: AsyncConnectionProto):
    cron_schedule = "* f f * *"  # invalid
    assistant = await client.assistants.create("agent")
    input = {"messages": [{"role": "human", "content": "foo"}]}
    with pytest.raises(httpx.HTTPStatusError):
        await client.crons.create(
            assistant["assistant_id"], schedule=cron_schedule, input=input
        )


async def test_cron_search(client: LangGraphClient, conn: AsyncConnectionProto):
    """
    Test cron search, try with multiple assistants
    """
    cron_schedule = "* * * * * *"  # every second
    assistant = await client.assistants.create("agent")
    assistant2 = await client.assistants.create("agent")
    inputs = {
        assistant["assistant_id"]: {"messages": [{"role": "human", "content": "foo"}]},
        assistant2["assistant_id"]: {"messages": [{"role": "human", "content": "bar"}]},
    }
    threads = {
        assistant["assistant_id"]: await client.threads.create(),
        assistant2["assistant_id"]: await client.threads.create(),
    }
    cron1 = await client.crons.create_for_thread(
        threads[assistant["assistant_id"]]["thread_id"],
        assistant["assistant_id"],
        schedule=cron_schedule,
        input=inputs[assistant["assistant_id"]],
    )
    cron2 = await client.crons.create_for_thread(
        threads[assistant2["assistant_id"]]["thread_id"],
        assistant2["assistant_id"],
        schedule=cron_schedule,
        input=inputs[assistant2["assistant_id"]],
    )
    await asyncio.sleep(2)

    cron_jobs = await client.crons.search()
    assert len(cron_jobs) == 2
    cron_jobs_first_assistant = await client.crons.search(
        assistant_id=assistant["assistant_id"]
    )
    assert len(cron_jobs_first_assistant) == 1
    assert cron_jobs_first_assistant[0]["schedule"] == "* * * * * *"

    cron_jobs_first_thread = await client.crons.search(
        assistant_id=assistant["assistant_id"],
        thread_id=threads[assistant["assistant_id"]]["thread_id"],
    )
    assert len(cron_jobs_first_thread) == 1

    await client.crons.delete(cron1["cron_id"])
    await client.crons.delete(cron2["cron_id"])

    cron_jobs = await client.crons.search(assistant_id=assistant["assistant_id"])
    assert len(cron_jobs) == 0


async def test_unicode_chars(client: LangGraphClient):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {
        "messages": [
            {"role": "human", "content": "\u0000hello-👋", "id": "initial-message"}
        ]
    }
    result = await client.runs.wait(
        thread["thread_id"], assistant["assistant_id"], input=input
    )
    assert result["messages"][0]["content"] == "hello-👋"
    assert result["some_bytes"] == b64encode(b"some_bytes").decode()
    assert result["some_byte_array"] == b64encode(b"some_byte_array").decode()
    assert result["dict_with_bytes"] == {
        "more_bytes": b64encode(b"more_bytes").decode()
    }


# test that metadata in langsmith run combines langgraph run, thread and assistant metadata


async def test_config_merging(client: LangGraphClient):
    model_name = "openai"
    temperature = 0.2
    assistant = await client.assistants.create(
        "agent", config={"configurable": {"model_name": model_name}}
    )
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    run = await client.runs.create(
        thread["thread_id"],
        assistant["assistant_id"],
        input=input,
        config={"configurable": {"temperature": temperature}},
    )
    assert run["kwargs"]["config"]["configurable"]["model_name"] == model_name
    assert run["kwargs"]["config"]["configurable"]["temperature"] == temperature


async def test_run_for_non_existent_assistant(client: LangGraphClient):
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}

    # test invalid assistant ID
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.runs.wait(thread["thread_id"], "fake-assistant-id", input=input)

    assert exc_info.value.response.status_code == 422

    # test a valid, but non-existent assistant ID
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.runs.wait(
            thread["thread_id"], "eb6db400-e3c8-5d06-a834-015cb89efe69", input=input
        )

    assert exc_info.value.response.status_code == 404


async def test_run_for_non_existent_thread(client: LangGraphClient):
    assistant = await client.assistants.create("agent")
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}

    # test invalid thread ID
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.runs.wait("fake-thread-id", assistant["assistant_id"], input=input)

    assert exc_info.value.response.status_code == 422

    # test a valid, but non-existent thread ID
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.runs.wait(
            "eb6db400-e3c8-5d06-a834-015cb89efe69",
            assistant["assistant_id"],
            input=input,
        )

    assert exc_info.value.response.status_code == 404


@pytest.mark.skip("This is flakey, due to langsmith polling, rethink this test")
async def test_langsmith_metadata(client: LangGraphClient):
    assistant_metadata = {"assistant_metadata": "assistant_metadata"}
    thread_metadata = {"thread_metadata": "thread_metadata"}
    run_metadata = {"run_metadata": "run_metadata"}

    assistant = await client.assistants.create("agent", metadata=assistant_metadata)
    thread = await client.threads.create(metadata=thread_metadata)
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    run = await client.runs.wait(
        thread["thread_id"],
        assistant["assistant_id"],
        input=input,
        metadata=run_metadata,
    )
    runs = await client.runs.list(thread["thread_id"])
    run = runs[0]

    # make sure the data is updated in langsmith
    await asyncio.sleep(5)
    # make sure all of the above metadata is combined in langsmith
    ls_client = LangSmithClient()
    run = ls_client.read_run(run["run_id"])
    assert (
        run.metadata["assistant_metadata"] == assistant_metadata["assistant_metadata"]
    )
    assert run.metadata["thread_metadata"] == thread_metadata["thread_metadata"]
    assert run.metadata["run_metadata"] == run_metadata["run_metadata"]


# Test multi-tasking


# TODO add tests for "wait" mode
RunMode: TypeAlias = Literal["background", "streaming"]
run_mode_params = pytest.mark.parametrize(
    ["run_1_mode", "run_2_mode"],
    [
        ["background", "background"],
        ["background", "streaming"],
        ["streaming", "background"],
        ["streaming", "streaming"],
    ],
)


@run_mode_params
async def test_multitasking_reject(
    client: LangGraphClient, run_1_mode: RunMode, run_2_mode: RunMode
):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}

    run_id = None
    if run_1_mode == "background":
        _run = await client.runs.create(
            thread["thread_id"], assistant["assistant_id"], input=input
        )
        run_id = _run["run_id"]
        # attempt another run to be rejected
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            if run_2_mode == "background":
                await client.runs.create(
                    thread["thread_id"],
                    assistant["assistant_id"],
                    input=input,
                    multitask_strategy="reject",
                )
            elif run_2_mode == "streaming":
                async for _ in client.runs.stream(
                    thread["thread_id"],
                    assistant["assistant_id"],
                    input=input,
                    multitask_strategy="reject",
                ):
                    pass
            else:
                raise NotImplementedError
    elif run_1_mode == "streaming":
        async for event in client.runs.stream(
            thread["thread_id"], assistant["assistant_id"], input=input
        ):
            if event.event == "metadata":
                run_id = event.data["run_id"]
                if run_2_mode == "background":
                    task = asyncio.create_task(
                        client.runs.create(
                            thread["thread_id"],
                            assistant["assistant_id"],
                            input=input,
                            multitask_strategy="reject",
                        )
                    )
                elif run_2_mode == "streaming":

                    async def stream():
                        async for _ in client.runs.stream(
                            thread["thread_id"],
                            assistant["assistant_id"],
                            input=input,
                            multitask_strategy="reject",
                        ):
                            pass

                    task = asyncio.create_task(stream())
                else:
                    raise NotImplementedError
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await task
    else:
        raise NotImplementedError

    assert exc_info.value.response.status_code == 409
    assert run_id is not None
    run_status = await poll_run(client, thread["thread_id"], run_id)
    assert run_status["status"] == "success"


@run_mode_params
async def test_multitasking_interrupt(
    client: LangGraphClient, run_1_mode: RunMode, run_2_mode: RunMode
):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input_1 = {
        "messages": [{"role": "human", "content": "foo", "id": "initial-message-1"}],
        "sleep": 3,
    }
    input_2 = {
        "messages": [{"role": "human", "content": "bar", "id": "initial-message-2"}]
    }

    if run_1_mode == "background":
        _run = await client.runs.create(
            thread["thread_id"], assistant["assistant_id"], input=input_1
        )
        await asyncio.sleep(1.5)
        run_id_1 = _run["run_id"]
        if run_2_mode == "background":
            _run = await client.runs.create(
                thread["thread_id"],
                assistant["assistant_id"],
                input=input_2,
                multitask_strategy="interrupt",
            )
            run_id_2 = _run["run_id"]
        elif run_2_mode == "streaming":
            async for event in client.runs.stream(
                thread["thread_id"],
                assistant["assistant_id"],
                input=input_2,
                multitask_strategy="interrupt",
            ):
                if event.event == "metadata":
                    run_id_2 = event.data["run_id"]
        else:
            raise NotImplementedError
    elif run_1_mode == "streaming":
        async for event in client.runs.stream(
            thread["thread_id"], assistant["assistant_id"], input=input_1
        ):
            if event.event == "metadata":
                run_id_1 = event.data["run_id"]
                if run_2_mode == "background":

                    async def schedule():
                        await asyncio.sleep(0.5)
                        _run = await client.runs.create(
                            thread["thread_id"],
                            assistant["assistant_id"],
                            input=input_2,
                            multitask_strategy="interrupt",
                        )
                        return _run["run_id"]

                    task = asyncio.create_task(schedule())
                elif run_2_mode == "streaming":

                    async def stream():
                        await asyncio.sleep(0.5)
                        async for event in client.runs.stream(
                            thread["thread_id"],
                            assistant["assistant_id"],
                            input=input_2,
                            multitask_strategy="interrupt",
                        ):
                            if event.event == "metadata":
                                run_id = event.data["run_id"]
                        return run_id

                    task = asyncio.create_task(stream())
                else:
                    raise NotImplementedError
        run_id_2 = await task
    else:
        raise NotImplementedError

    run_1_status = await poll_run(client, thread["thread_id"], run_id_1)
    assert run_1_status["status"] == "interrupted"
    run_2_status = await poll_run(client, thread["thread_id"], run_id_2)
    assert run_2_status["status"] == "success"

    state = await client.threads.get_state(thread["thread_id"])
    assert state["values"]["messages"][-4]["content"] == "bar"
    assert len(state["values"]["messages"]) in (5, 6, 7)
    assert state["values"]["messages"][0]["content"] == "foo"


@run_mode_params
async def test_multitasking_rollback(
    client: LangGraphClient, run_1_mode: RunMode, run_2_mode: RunMode
):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input_1 = {
        "messages": [{"role": "human", "content": "foo", "id": "initial-message-1"}],
        "sleep": 3,
    }
    input_2 = {
        "messages": [{"role": "human", "content": "bar", "id": "initial-message-2"}]
    }

    if run_1_mode == "background":
        _run = await client.runs.create(
            thread["thread_id"], assistant["assistant_id"], input=input_1
        )
        await asyncio.sleep(1.5)
        run_id_1 = _run["run_id"]
        if run_2_mode == "background":
            _run = await client.runs.create(
                thread["thread_id"],
                assistant["assistant_id"],
                input=input_2,
                multitask_strategy="rollback",
            )
            run_id_2 = _run["run_id"]
        elif run_2_mode == "streaming":
            async for event in client.runs.stream(
                thread["thread_id"],
                assistant["assistant_id"],
                input=input_2,
                multitask_strategy="rollback",
            ):
                if event.event == "metadata":
                    run_id_2 = event.data["run_id"]
        else:
            raise NotImplementedError
    elif run_1_mode == "streaming":
        async for event in client.runs.stream(
            thread["thread_id"], assistant["assistant_id"], input=input_1
        ):
            if event.event == "metadata":
                run_id_1 = event.data["run_id"]
                if run_2_mode == "background":

                    async def schedule():
                        _run = await client.runs.create(
                            thread["thread_id"],
                            assistant["assistant_id"],
                            input=input_2,
                            multitask_strategy="rollback",
                        )
                        return _run["run_id"]

                    task = asyncio.create_task(schedule())
                elif run_2_mode == "streaming":

                    async def stream():
                        async for event in client.runs.stream(
                            thread["thread_id"],
                            assistant["assistant_id"],
                            input=input_2,
                            multitask_strategy="rollback",
                        ):
                            if event.event == "metadata":
                                run_id = event.data["run_id"]
                        return run_id

                    task = asyncio.create_task(stream())
                else:
                    raise NotImplementedError
        run_id_2 = await task
    else:
        raise NotImplementedError

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await poll_run(client, thread["thread_id"], run_id_1)
    assert exc_info.value.response.status_code == 404
    run_2_status = await poll_run(client, thread["thread_id"], run_id_2)
    assert run_2_status["status"] == "success"

    state = await client.threads.get_state(thread["thread_id"])
    assert state["values"]["messages"][0]["content"] == "bar"
    assert len(state["values"]["messages"]) == 4


@run_mode_params
async def test_multitasking_enqueue(
    client: LangGraphClient, run_1_mode: RunMode, run_2_mode: RunMode
):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input_1 = {
        "messages": [{"role": "human", "content": "foo", "id": "initial-message-1"}],
        "sleep": 3,
    }
    input_2 = {
        "messages": [{"role": "human", "content": "bar", "id": "initial-message-2"}]
    }

    if run_1_mode == "background":
        _run = await client.runs.create(
            thread["thread_id"], assistant["assistant_id"], input=input_1
        )
        await asyncio.sleep(1.5)
        run_id_1 = _run["run_id"]
        if run_2_mode == "background":
            _run = await client.runs.create(
                thread["thread_id"],
                assistant["assistant_id"],
                input=input_2,
                multitask_strategy="enqueue",
            )
            run_id_2 = _run["run_id"]
        elif run_2_mode == "streaming":
            async for event in client.runs.stream(
                thread["thread_id"],
                assistant["assistant_id"],
                input=input_2,
                multitask_strategy="enqueue",
            ):
                if event.event == "metadata":
                    run_id_2 = event.data["run_id"]
        else:
            raise NotImplementedError
    elif run_1_mode == "streaming":
        async for event in client.runs.stream(
            thread["thread_id"], assistant["assistant_id"], input=input_1
        ):
            if event.event == "metadata":
                run_id_1 = event.data["run_id"]
                if run_2_mode == "background":

                    async def schedule():
                        _run = await client.runs.create(
                            thread["thread_id"],
                            assistant["assistant_id"],
                            input=input_2,
                            multitask_strategy="enqueue",
                        )
                        return _run["run_id"]

                    task = asyncio.create_task(schedule())
                elif run_2_mode == "streaming":

                    async def stream():
                        async for event in client.runs.stream(
                            thread["thread_id"],
                            assistant["assistant_id"],
                            input=input_2,
                            multitask_strategy="enqueue",
                        ):
                            if event.event == "metadata":
                                run_id = event.data["run_id"]
                        return run_id

                    task = asyncio.create_task(stream())
                else:
                    raise NotImplementedError
        run_id_2 = await task
    else:
        raise NotImplementedError

    run_1_status = await poll_run(client, thread["thread_id"], run_id_1)
    assert run_1_status["status"] == "success"
    run_2_status = await poll_run(client, thread["thread_id"], run_id_2)
    assert run_2_status["status"] == "success"

    state = await client.threads.get_state(thread["thread_id"])
    assert state["values"]["messages"][-4]["content"] == "bar"
    assert len(state["values"]["messages"]) == 8
    assert state["values"]["messages"][0]["content"] == "foo"


async def test_update_state(client: LangGraphClient):
    assistant = await client.assistants.create("agent")
    thread = await client.threads.create()
    input = {"messages": [{"role": "human", "content": "foo", "id": "initial-message"}]}
    run = await client.runs.create(
        thread["thread_id"], assistant["assistant_id"], input=input
    )

    await poll_run(client, thread["thread_id"], run["run_id"])

    history = await client.threads.get_history(thread["thread_id"])

    tool_event = history[1]
    assert tool_event["next"] == ["agent"]
    assert tool_event["metadata"]["step"] == 2
    assert tool_event["metadata"]["source"] == "loop"
    assert "tool" in tool_event["metadata"]["writes"]

    agent_event = history[2]
    assert agent_event["next"] == ["tool", "nothing"]
    assert agent_event["metadata"]["step"] == 1
    assert agent_event["metadata"]["source"] == "loop"
    assert "agent" in agent_event["metadata"]["writes"]

    await client.runs.wait(
        thread["thread_id"],
        assistant["assistant_id"],
        input=None,
        checkpoint_id=tool_event["checkpoint_id"],
    )

    new_history = await client.threads.get_history(thread["thread_id"])
    assert len(new_history) == len(history) + 1
    assert new_history[1:] == history

    # TODO assert

    from copy import deepcopy

    new_values = deepcopy(tool_event["metadata"]["writes"]["tool"])
    new_values["messages"][0]["content"] = "updated_tool_call__begin"

    await client.threads.update_state(
        thread_id=thread["thread_id"],
        values=new_values,
        checkpoint_id=agent_event["checkpoint_id"],
        # my assumption: I should act as the node who made the write
        as_node="tool",
    )
    history = await client.threads.get_history(thread["thread_id"])

    assert history[0]["metadata"]["step"] == 2
    assert history[0]["metadata"]["source"] == "update"
    assert "tool" in history[0]["metadata"]["writes"]
    assert history[0]["next"] == ["agent"]

    assert "tool" in history[0]["metadata"]["writes"]
    assert (
        history[0]["metadata"]["writes"]["tool"]["messages"][0]["content"]
        == "updated_tool_call__begin"
    )
    assert history[0]["values"]["messages"][-1]["content"] == "updated_tool_call__begin"
