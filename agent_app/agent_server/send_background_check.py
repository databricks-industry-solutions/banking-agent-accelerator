"""Send a background-check result to a waiting workflow thread.

Usage (local dev with Lakebase checkpointer):

    cd agent_app && python -m agent_server.send_background_check \
        --thread-id <thread-id> [--status approved|denied]

The script connects to the same Lakebase checkpointer the server uses,
builds the graph, and invokes it with background_check_result in the
input state.  This triggers the resume flow for the given thread.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import urllib.request
import json

from dotenv import load_dotenv


def _notify_chat_app(thread_id: str, chat_app_url: str) -> None:
    """Notify the chat app that a background-check result is available."""
    url = f"{chat_app_url.rstrip('/')}/api/internal/background-check-received"
    data = json.dumps({"chatId": thread_id}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"Chat app notified (HTTP {resp.status}).")
    except Exception as exc:
        print(f"WARNING: Could not notify chat app at {url}: {exc}", file=sys.stderr)


async def send_result(thread_id: str, status: str = "approved") -> None:
    load_dotenv(dotenv_path=".env", override=True)

    from databricks_langchain.checkpoint import AsyncCheckpointSaver

    from agent_server.langgraph_agent import build_graph

    instance_name = os.getenv("LAKEBASE_INSTANCE_NAME", "")
    if not instance_name:
        print("ERROR: LAKEBASE_INSTANCE_NAME env var is not set.", file=sys.stderr)
        sys.exit(1)

    bg_result = {
        "status": status,
        "details": f"Background check {status} (sent via CLI)",
    }

    async with AsyncCheckpointSaver(instance_name=instance_name) as checkpointer:
        await checkpointer.setup()
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        await graph.aupdate_state(
            config,
            {"background_check_result": bg_result},
        )

        state = await graph.aget_state(config)
        print(f"Thread:  {thread_id}")
        print(f"Status:  {status}")
        print(f"Stage:   {state.values.get('stage', '?')}")
        print("Background check result injected into checkpoint.")

        chat_app_port = os.getenv("CHAT_APP_PORT", "3000")
        chat_app_url = f"http://localhost:{chat_app_port}"
        _notify_chat_app(thread_id, chat_app_url)

        print("The next user message in the chat will trigger the resume flow.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deliver a background-check result to a waiting workflow thread.",
    )
    parser.add_argument("--thread-id", required=True, help="LangGraph thread ID to resume")
    parser.add_argument(
        "--status",
        default="approved",
        choices=["approved", "denied"],
        help="Background check outcome (default: approved)",
    )
    args = parser.parse_args()
    asyncio.run(send_result(args.thread_id, args.status))


if __name__ == "__main__":
    main()
