"""In-process smoke tests for the banking workflow state machine.

Run:  cd agent_app && python -m agent_server.dev_smoke_test

Uses MemorySaver (no Lakebase / Databricks credentials required).
"""

from __future__ import annotations

import asyncio
import sys

from langgraph.checkpoint.memory import MemorySaver

from agent_server.langgraph_agent import build_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_msg(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


async def _inject_background_check(graph, config, status="approved"):
    """Inject a background-check result into a WAITING thread and return new state."""
    return await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": f"[BACKGROUND_CHECK_RESULT: {status}]"}],
            "background_check_result": {
                "status": status,
                "details": f"Background check {status}",
            },
        },
        config,
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


async def test_happy_path() -> None:
    """Full account-statement flow: classify -> fields -> bg check -> confirm -> send."""
    print("=== Happy path ===")
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "happy-1"}}

    # Turn 1: intent
    state = await graph.ainvoke(
        {**_user_msg("Generate account statement"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "EXTRACT_FIELDS", f"Expected EXTRACT_FIELDS, got {state['stage']}"
    assert state["intent"] == "GENERATE_ACCOUNT_STATEMENT"
    print(f"  Turn 1  stage={state['stage']}")

    # Turn 2: provide all fields -> background check submitted, graph pauses
    state = await graph.ainvoke(
        {
            **_user_msg(
                "customer_id=C001 account_id=A100 period_start=2025-01-01 period_end=2025-12-31"
            ),
            "stub_scenario": "happy_path",
        },
        config,
    )
    assert state["stage"] == "WAITING_FOR_BACKGROUND_CHECK", (
        f"Expected WAITING_FOR_BACKGROUND_CHECK, got {state['stage']}"
    )
    assert state["background_check_request_id"], "Must have a background_check_request_id"
    print(f"  Turn 2  stage={state['stage']}  request_id={state['background_check_request_id']}")

    # Turn 3: inject background check result -> resumes to SEND_EMAIL
    state = await _inject_background_check(graph, config)
    assert state["stage"] == "SEND_EMAIL", f"Expected SEND_EMAIL, got {state['stage']}"
    last_ai_msg = [m for m in state["messages"] if hasattr(m, "type") and m.type == "ai"][-1]
    assert "SEND" in last_ai_msg.content, "Preview message must prompt user to reply SEND"
    assert state["email_subject"] in last_ai_msg.content, "Preview must include email subject"
    print(f"  Turn 3  stage={state['stage']} (preview emitted by confirm_node)")

    # Turn 4: confirm
    state = await graph.ainvoke(
        {**_user_msg("SEND"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "DONE", f"Expected DONE, got {state['stage']}"
    print(f"  Turn 4  stage={state['stage']}")
    print("  PASSED\n")


async def test_missing_fields_loop() -> None:
    """Provide partial fields, verify re-ask, then provide the rest."""
    print("=== Missing fields loop ===")
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "missing-1"}}

    # Turn 1: intent
    state = await graph.ainvoke(
        {**_user_msg("I need an account statement"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "EXTRACT_FIELDS"
    assert len(state["missing_fields"]) == 4
    print(f"  Turn 1  stage={state['stage']}  missing={state['missing_fields']}")

    # Turn 2: partial -- only customer_id and account_id
    state = await graph.ainvoke(
        {**_user_msg("customer_id=C002 account_id=A200"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "EXTRACT_FIELDS", f"Expected EXTRACT_FIELDS, got {state['stage']}"
    assert set(state["missing_fields"]) == {"period_start", "period_end"}
    print(f"  Turn 2  stage={state['stage']}  missing={state['missing_fields']}")

    # Turn 3: provide remaining fields -> waiting for background check
    state = await graph.ainvoke(
        {
            **_user_msg("period_start=2025-01-01 period_end=2025-06-30"),
            "stub_scenario": "happy_path",
        },
        config,
    )
    assert state["stage"] == "WAITING_FOR_BACKGROUND_CHECK", (
        f"Expected WAITING_FOR_BACKGROUND_CHECK, got {state['stage']}"
    )
    assert not state["missing_fields"]
    print(f"  Turn 3  stage={state['stage']}  missing={state['missing_fields']}")

    # Turn 4: inject background check result
    state = await _inject_background_check(graph, config)
    assert state["stage"] == "SEND_EMAIL", f"Expected SEND_EMAIL, got {state['stage']}"
    print(f"  Turn 4  stage={state['stage']}")

    # Turn 5: confirm
    state = await graph.ainvoke(
        {**_user_msg("SEND"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "DONE", f"Expected DONE, got {state['stage']}"
    print(f"  Turn 5  stage={state['stage']}")
    print("  PASSED\n")


async def test_confirmation_gating() -> None:
    """After preview, send a field change instead of SEND -- must NOT send."""
    print("=== Confirmation gating ===")
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "gate-1"}}

    # Turn 1: intent
    state = await graph.ainvoke(
        {**_user_msg("Generate account statement"), "stub_scenario": "happy_path"},
        config,
    )

    # Turn 2: all fields -> waiting
    state = await graph.ainvoke(
        {
            **_user_msg(
                "customer_id=C003 account_id=A300 period_start=2025-01-01 period_end=2025-12-31"
            ),
            "stub_scenario": "happy_path",
        },
        config,
    )
    assert state["stage"] == "WAITING_FOR_BACKGROUND_CHECK"

    # Turn 3: inject background check result -> SEND_EMAIL
    state = await _inject_background_check(graph, config)
    assert state["stage"] == "SEND_EMAIL"
    print(f"  Turn 3  stage={state['stage']} (preview shown)")

    # Turn 4: change a field instead of confirming -> triggers new background check
    state = await graph.ainvoke(
        {**_user_msg("period_end=2025-06-30"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] != "DONE", "Graph must NOT send when user specifies changes"
    assert state["field_values"]["period_end"] == "2025-06-30"
    print(
        f"  Turn 4  stage={state['stage']}  period_end={state['field_values']['period_end']}"
        f"  (change accepted, not sent)"
    )

    # Turn 5: inject background check result again
    state = await _inject_background_check(graph, config)
    assert state["stage"] == "SEND_EMAIL"

    # Turn 6: now confirm
    state = await graph.ainvoke(
        {**_user_msg("SEND"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "DONE", f"Expected DONE, got {state['stage']}"
    print(f"  Turn 6  stage={state['stage']}")
    print("  PASSED\n")


async def test_natural_confirm_words() -> None:
    """Expanded confirm words like 'go ahead' and 'ok' must also trigger send."""
    print("=== Natural confirm words ===")

    for word in ("go ahead", "ok", "sure", "looks good", "do it"):
        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": f"confirm-{word}"}}

        # Turn 1: intent
        await graph.ainvoke(
            {**_user_msg("Generate account statement"), "stub_scenario": "happy_path"},
            config,
        )

        # Turn 2: all fields -> waiting
        state = await graph.ainvoke(
            {
                **_user_msg(
                    "customer_id=C010 account_id=A110 "
                    "period_start=2025-01-01 period_end=2025-12-31"
                ),
                "stub_scenario": "happy_path",
            },
            config,
        )
        assert state["stage"] == "WAITING_FOR_BACKGROUND_CHECK", (
            f"[{word}] Expected WAITING_FOR_BACKGROUND_CHECK, got {state['stage']}"
        )

        # Turn 3: inject background check result -> SEND_EMAIL
        state = await _inject_background_check(graph, config)
        assert state["stage"] == "SEND_EMAIL", (
            f"[{word}] Expected SEND_EMAIL, got {state['stage']}"
        )

        # Turn 4: confirm with natural language
        state = await graph.ainvoke(
            {**_user_msg(word), "stub_scenario": "happy_path"},
            config,
        )
        assert state["stage"] == "DONE", (
            f"[{word}] Expected DONE after '{word}', got {state['stage']}"
        )
        print(f"  '{word}' -> stage={state['stage']}")

    print("  PASSED\n")


async def test_change_customer_id_at_send_email() -> None:
    """Changing customer_id at SEND_EMAIL must re-lookup email and update preview."""
    print("=== Change customer_id at SEND_EMAIL ===")
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "change-cid-1"}}

    # Turn 1: intent
    await graph.ainvoke(
        {**_user_msg("Generate account statement"), "stub_scenario": "happy_path"},
        config,
    )

    # Turn 2: all fields -> waiting
    state = await graph.ainvoke(
        {
            **_user_msg(
                "customer_id=C001 account_id=A100 "
                "period_start=2025-01-01 period_end=2025-12-31"
            ),
            "stub_scenario": "happy_path",
        },
        config,
    )
    assert state["stage"] == "WAITING_FOR_BACKGROUND_CHECK"

    # Turn 3: inject bg result -> SEND_EMAIL
    state = await _inject_background_check(graph, config)
    assert state["stage"] == "SEND_EMAIL"
    assert state["email_to"] == "C001@example.com"
    print(f"  Turn 3  stage={state['stage']}  email_to={state['email_to']}")

    # Turn 4: change customer_id -> re-lookup -> new background check
    state = await graph.ainvoke(
        {**_user_msg("customer_id=C999"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "WAITING_FOR_BACKGROUND_CHECK", (
        f"Expected WAITING_FOR_BACKGROUND_CHECK, got {state['stage']}"
    )
    assert state["field_values"]["customer_id"] == "C999"
    assert state["email_to"] == "C999@example.com", (
        f"Expected re-looked-up email C999@example.com, got {state['email_to']}"
    )
    print(f"  Turn 4  stage={state['stage']}  email_to={state['email_to']} (re-resolved)")

    # Turn 5: inject bg result again -> SEND_EMAIL
    state = await _inject_background_check(graph, config)
    assert state["stage"] == "SEND_EMAIL"

    # Turn 6: confirm
    state = await graph.ainvoke(
        {**_user_msg("SEND"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "DONE"
    print(f"  Turn 6  stage={state['stage']}")
    print("  PASSED\n")


async def test_open_deposit_incremental_fields() -> None:
    """OPEN_DEPOSIT: provide fields incrementally and reach email preview."""
    print("=== Open deposit incremental fields ===")
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "deposit-1"}}

    # Turn 1: intent
    state = await graph.ainvoke(
        {**_user_msg("I want to open a deposit"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "EXTRACT_FIELDS", f"Expected EXTRACT_FIELDS, got {state['stage']}"
    assert state["intent"] == "OPEN_DEPOSIT"
    assert len(state["missing_fields"]) == 5
    print(f"  Turn 1  stage={state['stage']}  missing={state['missing_fields']}")

    # Turn 2: partial -- customer_id, amount, currency
    state = await graph.ainvoke(
        {
            **_user_msg("customer_id=C100 amount=5000 currency=EUR"),
            "stub_scenario": "happy_path",
        },
        config,
    )
    assert state["stage"] == "EXTRACT_FIELDS", f"Expected EXTRACT_FIELDS, got {state['stage']}"
    assert set(state["missing_fields"]) == {"term_months", "payout_account"}
    print(f"  Turn 2  stage={state['stage']}  missing={state['missing_fields']}")

    # Turn 3: provide remaining fields -> waiting
    state = await graph.ainvoke(
        {
            **_user_msg("term_months=12 payout_account=ACC999"),
            "stub_scenario": "happy_path",
        },
        config,
    )
    assert state["stage"] == "WAITING_FOR_BACKGROUND_CHECK", (
        f"Expected WAITING_FOR_BACKGROUND_CHECK, got {state['stage']}"
    )
    assert not state["missing_fields"]
    print(f"  Turn 3  stage={state['stage']}  (waiting for background check)")

    # Turn 4: inject background check result -> SEND_EMAIL
    state = await _inject_background_check(graph, config)
    assert state["stage"] == "SEND_EMAIL", f"Expected SEND_EMAIL, got {state['stage']}"
    last_ai_msg = [m for m in state["messages"] if hasattr(m, "type") and m.type == "ai"][-1]
    assert "SEND" in last_ai_msg.content, "Preview message must prompt user to reply SEND"
    print(f"  Turn 4  stage={state['stage']}  (email preview shown)")

    # Turn 5: confirm
    state = await graph.ainvoke(
        {**_user_msg("SEND"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "DONE", f"Expected DONE, got {state['stage']}"
    print(f"  Turn 5  stage={state['stage']}")
    print("  PASSED\n")


async def test_change_non_customer_field_at_send_email() -> None:
    """Changing a field other than customer_id at SEND_EMAIL must update the preview."""
    print("=== Change non-customer_id field at SEND_EMAIL ===")
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "change-amount-1"}}

    # Turn 1: intent
    await graph.ainvoke(
        {**_user_msg("I want to open a deposit"), "stub_scenario": "happy_path"},
        config,
    )

    # Turn 2: all fields -> waiting
    state = await graph.ainvoke(
        {
            **_user_msg(
                "customer_id=C100 amount=5000 currency=EUR "
                "term_months=12 payout_account=ACC999"
            ),
            "stub_scenario": "happy_path",
        },
        config,
    )
    assert state["stage"] == "WAITING_FOR_BACKGROUND_CHECK"

    # Turn 3: inject bg result -> SEND_EMAIL
    state = await _inject_background_check(graph, config)
    assert state["stage"] == "SEND_EMAIL"
    assert state["field_values"]["amount"] == "5000"
    print(f"  Turn 3  stage={state['stage']}  amount={state['field_values']['amount']}")

    # Turn 4: change amount (not customer_id) -> goes through lookup again -> new bg check
    state = await graph.ainvoke(
        {**_user_msg("amount=10000"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "WAITING_FOR_BACKGROUND_CHECK", (
        f"Expected WAITING_FOR_BACKGROUND_CHECK, got {state['stage']}"
    )
    assert state["field_values"]["amount"] == "10000", (
        f"Expected amount=10000, got {state['field_values']['amount']}"
    )
    print(
        f"  Turn 4  stage={state['stage']}  amount={state['field_values']['amount']}"
        f"  (change accepted, new background check)"
    )

    # Turn 5: inject bg result -> SEND_EMAIL with updated preview
    state = await _inject_background_check(graph, config)
    assert state["stage"] == "SEND_EMAIL"
    last_ai_msg = [m for m in state["messages"] if hasattr(m, "type") and m.type == "ai"][-1]
    assert "10000" in last_ai_msg.content, "Updated amount must appear in preview"
    print(f"  Turn 5  stage={state['stage']}  (preview updated with 10000)")

    # Turn 6: confirm
    state = await graph.ainvoke(
        {**_user_msg("SEND"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "DONE", f"Expected DONE, got {state['stage']}"
    print(f"  Turn 6  stage={state['stage']}")
    print("  PASSED\n")


async def test_background_check_happy_path() -> None:
    """Verify the full async flow: submit -> wait -> inject result -> confirm."""
    print("=== Background check happy path ===")
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "bgcheck-1"}}

    # Turn 1: intent
    state = await graph.ainvoke(
        {**_user_msg("Generate account statement"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "EXTRACT_FIELDS"
    print(f"  Turn 1  stage={state['stage']}")

    # Turn 2: all fields -> background check submitted
    state = await graph.ainvoke(
        {
            **_user_msg(
                "customer_id=C050 account_id=A500 "
                "period_start=2025-01-01 period_end=2025-12-31"
            ),
            "stub_scenario": "happy_path",
        },
        config,
    )
    assert state["stage"] == "WAITING_FOR_BACKGROUND_CHECK"
    assert state["background_check_request_id"]
    assert state["email_to"] == "C050@example.com"
    print(
        f"  Turn 2  stage={state['stage']}  "
        f"request_id={state['background_check_request_id']}"
    )

    # Turn 3: inject approved result -> resume -> confirm -> SEND_EMAIL
    state = await _inject_background_check(graph, config, status="approved")
    assert state["stage"] == "SEND_EMAIL", f"Expected SEND_EMAIL, got {state['stage']}"
    assert state["background_check_result"] is None, "Result must be cleared after resume"
    print(f"  Turn 3  stage={state['stage']}  (resumed, preview shown)")

    # Turn 4: confirm
    state = await graph.ainvoke(
        {**_user_msg("SEND"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "DONE"
    print(f"  Turn 4  stage={state['stage']}")
    print("  PASSED\n")


async def test_background_check_waiting_blocks_user() -> None:
    """User messages while WAITING must not advance the workflow stage."""
    print("=== Background check waiting blocks user ===")
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "bgcheck-block-1"}}

    # Turn 1: intent
    await graph.ainvoke(
        {**_user_msg("Generate account statement"), "stub_scenario": "happy_path"},
        config,
    )

    # Turn 2: all fields -> waiting
    state = await graph.ainvoke(
        {
            **_user_msg(
                "customer_id=C060 account_id=A600 "
                "period_start=2025-01-01 period_end=2025-12-31"
            ),
            "stub_scenario": "happy_path",
        },
        config,
    )
    assert state["stage"] == "WAITING_FOR_BACKGROUND_CHECK"
    print(f"  Turn 2  stage={state['stage']}")

    # Turn 3-5: user sends various messages -- stage must NOT change
    for msg_text in ("What's happening?", "SEND", "customer_id=C999"):
        state = await graph.ainvoke(
            {**_user_msg(msg_text), "stub_scenario": "happy_path"},
            config,
        )
        assert state["stage"] == "WAITING_FOR_BACKGROUND_CHECK", (
            f"Stage must remain WAITING_FOR_BACKGROUND_CHECK after '{msg_text}', "
            f"got {state['stage']}"
        )
        print(f"  '{msg_text}' -> stage={state['stage']}  (blocked)")

    # Turn 6: inject result -> resumes normally
    state = await _inject_background_check(graph, config)
    assert state["stage"] == "SEND_EMAIL", f"Expected SEND_EMAIL, got {state['stage']}"
    print(f"  inject result -> stage={state['stage']}  (resumed)")
    print("  PASSED\n")


async def test_background_check_denied() -> None:
    """Denied background check must terminate the workflow permanently."""
    print("=== Background check denied ===")
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "bgcheck-denied-1"}}

    # Turn 1: intent
    state = await graph.ainvoke(
        {**_user_msg("Generate account statement"), "stub_scenario": "happy_path"},
        config,
    )
    assert state["stage"] == "EXTRACT_FIELDS"
    print(f"  Turn 1  stage={state['stage']}")

    # Turn 2: all fields -> waiting
    state = await graph.ainvoke(
        {
            **_user_msg(
                "customer_id=C070 account_id=A700 "
                "period_start=2025-01-01 period_end=2025-12-31"
            ),
            "stub_scenario": "happy_path",
        },
        config,
    )
    assert state["stage"] == "WAITING_FOR_BACKGROUND_CHECK"
    print(f"  Turn 2  stage={state['stage']}")

    # Turn 3: inject denied result -> DENIED
    state = await _inject_background_check(graph, config, status="denied")
    assert state["stage"] == "DENIED", f"Expected DENIED, got {state['stage']}"
    assert state["background_check_result"] is None, "Result must be cleared after resume"
    last_ai = [m for m in state["messages"] if hasattr(m, "type") and m.type == "ai"][-1]
    assert "denied" in last_ai.content.lower(), "Denial message must mention 'denied'"
    print(f"  Turn 3  stage={state['stage']}  (denied, workflow terminated)")

    # Turns 4-6: any user input must NOT change the stage
    for msg_text in ("What happened?", "SEND", "customer_id=C999"):
        state = await graph.ainvoke(
            {**_user_msg(msg_text), "stub_scenario": "happy_path"},
            config,
        )
        assert state["stage"] == "DENIED", (
            f"Stage must remain DENIED after '{msg_text}', got {state['stage']}"
        )
        print(f"  '{msg_text}' -> stage={state['stage']}  (blocked)")

    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run_all() -> None:
    await test_happy_path()
    await test_missing_fields_loop()
    await test_confirmation_gating()
    await test_natural_confirm_words()
    await test_change_customer_id_at_send_email()
    await test_open_deposit_incremental_fields()
    await test_change_non_customer_field_at_send_email()
    await test_background_check_happy_path()
    await test_background_check_waiting_blocks_user()
    await test_background_check_denied()
    print("All smoke tests passed.")


if __name__ == "__main__":
    try:
        asyncio.run(_run_all())
    except AssertionError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        sys.exit(1)
