import asyncio  # noqa: I001
from typing import Annotated, Any, Sequence, TypedDict

import httpx
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.managed.shared_value import SharedValue
from langgraph.constants import Send
from langgraph_sdk import get_client

sdk = get_client()


class AgentState(TypedDict):
    some_bytes: bytes
    some_byte_array: bytearray
    dict_with_bytes: dict[str, bytes]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sleep: int
    shared: Annotated[dict[str, dict[str, Any]], SharedValue.on("assistant_id")]
    expect_shared_value: bool


async def call_model(state, config):
    if sleep := state.get("sleep"):
        await asyncio.sleep(sleep)
    if expect_shared_value := state.get("expect_shared_value"):
        assert state["shared"] == {"key": {"my": "value"}}, state["shared"]

    messages = state["messages"]

    if len(messages) > 1 and any(not isinstance(m, HumanMessage) for m in messages):
        assert state["some_bytes"] == b"some_bytes", state["some_bytes"]
        assert state["some_byte_array"] == bytearray(b"some_byte_array"), state[
            "some_byte_array"
        ]
        assert state["dict_with_bytes"] == {"more_bytes": b"more_bytes"}, state[
            "dict_with_bytes"
        ]

    # hacky way to reset model to the "first" response
    if isinstance(messages[-1], HumanMessage):
        model.i = 0

    response = await model.ainvoke(messages)
    final = {
        "messages": [response],
        "some_bytes": b"some_bytes",
        "some_byte_array": bytearray(b"some_byte_array"),
        "dict_with_bytes": {"more_bytes": b"more_bytes"},
    }
    if not expect_shared_value:
        final["shared"] = {"key": {"my": "value"}}
    return final


async def call_tool(last_message, config):
    assert isinstance(sdk.http.client._transport, httpx.ASGITransport)
    assert (
        len(await sdk.assistants.search(graph_id=config["configurable"]["graph_id"]))
        > 0
    )
    return {
        "messages": [
            ToolMessage(
                f"tool_call__{last_message.content}", tool_call_id="tool_call_id"
            )
        ]
    }


def do_nothing(last_message):
    pass


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.content == "end":
        return END
    else:
        return [Send("tool", last_message), Send("nothing", last_message)]


# NOTE: the model cycles through responses infinitely here
model = FakeListChatModel(responses=["begin", "end"])
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tool", call_tool)
workflow.add_node("nothing", do_nothing)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("tool", "agent")

graph = workflow.compile()
