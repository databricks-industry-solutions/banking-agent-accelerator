import asyncio
import logging
import os
import uuid
from typing import AsyncGenerator, Optional

import mlflow
from databricks_langchain.chat_models import ChatDatabricks
from databricks_langchain.checkpoint import AsyncCheckpointSaver
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

import json
import urllib.request

from agent_server.langgraph_agent import build_graph
from agent_server.utils import process_agent_astream_events

logger = logging.getLogger(__name__)


def _notify_chat_app(thread_id: str) -> None:
    """Best-effort notification to the chat frontend (same as send_background_check.py)."""
    port = os.getenv("CHAT_APP_PORT", "3000")
    url = f"http://localhost:{port}/api/internal/background-check-received"
    data = json.dumps({"chatId": thread_id}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            logger.info("Chat app notified (HTTP %s).", resp.status)
    except Exception as exc:
        logger.debug("Could not notify chat app at %s: %s", url, exc)


mlflow.langchain.autolog()
_llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4")
LAKEBASE_INSTANCE_NAME = os.getenv("LAKEBASE_INSTANCE_NAME", "")
_CHECKPOINTER_SETUP_DONE = False
_CHECKPOINTER_SETUP_LOCK: Optional[asyncio.Lock] = None

if not LAKEBASE_INSTANCE_NAME:
    raise ValueError(
        "LAKEBASE_INSTANCE_NAME environment variable is required but not set. "
        "Please set it in your environment or in `agent_app/.env`."
    )


async def _ensure_checkpointer_setup(checkpointer: AsyncCheckpointSaver) -> None:
    global _CHECKPOINTER_SETUP_DONE, _CHECKPOINTER_SETUP_LOCK

    if _CHECKPOINTER_SETUP_DONE:
        return

    if _CHECKPOINTER_SETUP_LOCK is None:
        _CHECKPOINTER_SETUP_LOCK = asyncio.Lock()

    async with _CHECKPOINTER_SETUP_LOCK:
        if _CHECKPOINTER_SETUP_DONE:
            return
        await checkpointer.setup()
        _CHECKPOINTER_SETUP_DONE = True


def _get_or_create_thread_id(request: ResponsesAgentRequest) -> str:
    custom_inputs = dict(request.custom_inputs or {})

    if custom_inputs.get("thread_id"):
        return str(custom_inputs["thread_id"])

    if request.context and getattr(request.context, "conversation_id", None):
        return str(request.context.conversation_id)

    return str(uuid.uuid4())


@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    thread_id = _get_or_create_thread_id(request)
    request.custom_inputs = dict(request.custom_inputs or {})
    request.custom_inputs["thread_id"] = thread_id

    outputs = [
        event.item
        async for event in streaming(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs, custom_outputs={"thread_id": thread_id})


@stream()
async def streaming(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    thread_id = _get_or_create_thread_id(request)
    mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})

    custom_inputs = dict(request.custom_inputs or {})
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {
        "messages": to_chat_completions_input([i.model_dump() for i in request.input]),
        "stub_scenario": custom_inputs.get("stub_scenario", "happy_path"),
    }
    bg_result = custom_inputs.get("background_check_result")

    if bg_result:
        async with AsyncCheckpointSaver(instance_name=LAKEBASE_INSTANCE_NAME) as checkpointer:
            await _ensure_checkpointer_setup(checkpointer)
            graph = build_graph(checkpointer=checkpointer, llm=_llm)
            await graph.aupdate_state(config, {"background_check_result": bg_result})
        await asyncio.to_thread(_notify_chat_app, thread_id)
        return

    async with AsyncCheckpointSaver(instance_name=LAKEBASE_INSTANCE_NAME) as checkpointer:
        await _ensure_checkpointer_setup(checkpointer)
        graph = build_graph(checkpointer=checkpointer, llm=_llm)
        async for event in process_agent_astream_events(
            graph.astream(input_state, config, stream_mode=["updates", "messages"])
        ):
            yield event

        try:
            state = await graph.aget_state(config)
            values = state.values
            field_values = values.get("field_values") or {}
            yield ResponsesAgentStreamEvent(
                type="workflow.state.updated",
                custom_outputs={
                    "workflow_stage": values.get("stage", ""),
                    "workflow_intent": values.get("intent", ""),
                    "workflow_customer_name": field_values.get("customer_id", ""),
                },
            )
        except Exception:
            logger.exception("Failed to emit workflow metadata event")
