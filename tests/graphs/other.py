import asyncio  # noqa: I001
from typing import Annotated, Literal, Optional, Sequence, TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langgraph.graph import END, StateGraph, add_messages
from langchain_core.pydantic_v1 import BaseModel


class SomeOtherModel(BaseModel):
    yo: int


def make_graph(config):
    class AgentState(BaseModel):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        other_model: SomeOtherModel
        sleep: Optional[int] = None

    class Config(TypedDict):
        model: Literal["openai", "anthropic"]

    async def call_model(state: AgentState, config: Config):
        assert isinstance(state, AgentState)
        assert isinstance(state, BaseModel)
        assert isinstance(state.other_model, SomeOtherModel)
        assert isinstance(state.other_model, BaseModel)

        if state.sleep:
            await asyncio.sleep(state.sleep)

        messages = state.messages

        # hacky way to reset model to the "first" response
        if isinstance(messages[-1], HumanMessage):
            model.i = 0

        response = await model.ainvoke(messages)
        return {
            "messages": [response],
        }

    def call_tool(state: AgentState):
        last_message_content = state.messages[-1].content
        return {
            "messages": [
                ToolMessage(
                    f"tool_call__{last_message_content}", tool_call_id="tool_call_id"
                )
            ]
        }

    def should_continue(state: AgentState):
        messages = state.messages
        last_message = messages[-1]
        if last_message.content == "end":
            return END
        else:
            return "tool"

    # NOTE: the model cycles through responses infinitely here
    model = FakeListChatModel(responses=["begin", "end"])
    workflow = StateGraph(AgentState, Config)

    workflow.add_node("agent", call_model)
    workflow.add_node("tool", call_tool)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )

    workflow.add_edge("tool", "agent")

    return workflow.compile()
