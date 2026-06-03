"""LangGraph deterministic state machine for the banking workflow.

The graph routes strictly by ``stage`` -- the LLM never decides next steps.
Stub tools handle get_template, lookup_email, render_email, send_email.
LLM handles intent classification, field extraction, message classification,
and free-form questions (with stub fallback when no LLM is provided, for testing).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent_server.tools_stubs import (
    classify_intent,
    extract_fields,
    field_label,
    get_template,
    lookup_customer_email,
    render_email,
    send_email,
    submit_background_check,
)

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]
    stage: str
    intent: str
    required_fields: list[str]
    field_values: dict[str, str]
    missing_fields: list[str]
    template_id: Optional[str]
    template_body: Optional[str]
    email_to: Optional[str]
    email_subject: Optional[str]
    email_body: Optional[str]
    email_candidates: list[str]
    last_error: Optional[dict]
    retry_node: Optional[str]
    stub_scenario: str
    is_question: bool
    background_check_request_id: Optional[str]
    background_check_result: Optional[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INTENT_LABELS: dict[str, str] = {
    "ADD_BENEFICIARY": "Add Beneficiary",
    "REQUEST_CREDIT_LIMIT_INCREASE": "Credit Limit Increase",
}

_STAGE_STEP: dict[str, int] = {
    "START": 1,
    "CLASSIFY_INTENT": 1,
    "GET_TEMPLATE": 2,
    "ASK_FOR_FIELDS": 3,
    "EXTRACT_FIELDS": 3,
    "LOOKUP_CUSTOMER_EMAIL": 4,
    "CUSTOMER_BACKGROUND_CHECK": 5,
    "WAITING_FOR_BACKGROUND_CHECK": 5,
    "DENIED": 5,
    "CONFIRM": 6,
    "SEND_EMAIL": 7,
    "DONE": 8,
}

_TOTAL_STEPS = 8

_STEP_MESSAGES: dict[int, str] = {
    1: f"**Step 1/{_TOTAL_STEPS}** — Identifying your request…\n\n",
    2: f"**Step 2/{_TOTAL_STEPS}** — Loading template…\n\n",
    3: f"**Step 3/{_TOTAL_STEPS}** — Collecting required information…\n\n",
    4: f"**Step 4/{_TOTAL_STEPS}** — Looking up customer email…\n\n",
    5: f"**Step 5/{_TOTAL_STEPS}** — Running background check…\n\n",
    6: f"**Step 6/{_TOTAL_STEPS}** — Preparing email preview…\n\n",
    7: f"**Step 7/{_TOTAL_STEPS}** — Review and confirm…\n\n",
    8: f"**Step 8/{_TOTAL_STEPS}** — Complete!\n\n",
}


def _step_entry_messages(old_stage: str, new_stage: str) -> list[AIMessage]:
    """Return a list with a single step-entry AIMessage if the step changed."""
    old_step = _STAGE_STEP.get(old_stage, 0)
    new_step = _STAGE_STEP.get(new_stage, 0)
    if new_step != old_step and new_step in _STEP_MESSAGES:
        return [AIMessage(content=_STEP_MESSAGES[new_step], name="step_notification")]
    return []


_CONFIRM_WORDS = frozenset({
    "send", "yes send", "yes", "confirm",
    "go ahead", "ok", "okay", "sure", "do it",
    "please send", "send it", "yes please",
    "go", "approved", "lgtm", "looks good",
})

_STAGE_EXPECTATIONS: dict[str, str] = {
    "EXTRACT_FIELDS": "field values for the banking workflow (e.g. customer_id, beneficiary or card details, amounts)",
    "SEND_EMAIL": "explicit confirmation to send (e.g. 'SEND', 'yes') or field changes",
    "WAITING_FOR_BACKGROUND_CHECK": "no user input expected; waiting for an external background check result",
    "DENIED": "the background check was denied; the workflow is permanently terminated",
    "ERROR": "'retry' to retry the failed operation, or 'cancel' to abort",
    "DONE": "the workflow is complete; no further input is expected",
}

_STAGE_INSTRUCTIONS: dict[str, str] = {
    "EXTRACT_FIELDS": (
        "The user needs to provide missing field values. A status checklist "
        "with exact field labels is already shown above your response. "
        "Refer to the missing fields using the EXACT labels from that "
        "checklist — do NOT rephrase, paraphrase, or invent your own names. "
        "If there are email candidates listed in the current state, ask the "
        "user to choose one. Do NOT confirm, summarize, or declare the "
        "request complete — fields are still missing. Your ONLY job right "
        "now is to ask for them."
    ),
    "SEND_EMAIL": (
        "You MUST display the full email preview exactly as shown in the current "
        "state (include both the subject line and the body). Then ask the user to "
        "reply SEND to confirm, or to describe any changes they want. "
        "NEVER say you will process or send the email yourself."
    ),
    "DONE": (
        "Confirm that the email was sent successfully. "
        "Let the user know this conversation is now complete. "
        "If they need help with anything else, they should start a new conversation. "
        "Do NOT offer to help with additional tasks in this thread."
    ),
    "ERROR": (
        "Explain what went wrong using the error details from the current state. "
        "Let the user know they can reply 'retry' to try again or 'cancel' to abort."
    ),
    "CLASSIFY_INTENT": (
        "The user's request does not match any supported operation. "
        "You can ONLY help with: (1) adding a new beneficiary / payee, or "
        "(2) requesting a credit limit increase. You CANNOT help with anything "
        "else — do NOT pretend you can handle unsupported requests such as "
        "money transfers, payments, loan applications, or any other banking "
        "operation. Politely explain what you CAN do and ask the user to choose."
    ),
    "WAITING_FOR_BACKGROUND_CHECK": (
        "A mandatory background check is currently in progress. The user "
        "must wait for it to complete before the workflow can continue. "
        "Politely inform them that the check is being processed and they "
        "will be notified automatically when results are ready. Do NOT "
        "process any workflow actions or offer to skip this step."
    ),
    "DENIED": (
        "The background check was denied and this workflow is permanently "
        "terminated. Inform the user that the process cannot continue and "
        "they must start a new chat if they wish to begin a new request. "
        "Do NOT offer to help with anything else in this thread."
    ),
}


def _get_latest_user_text(state: WorkflowState) -> str | None:
    """Return the content of the most recent user / human message."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                return msg.get("content", "")
        elif hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return None


def _format_status_preamble(state: WorkflowState) -> str:
    """Deterministic field checklist shown at the top of every user-facing message."""
    required = state.get("required_fields", [])
    values = state.get("field_values", {})
    lines: list[str] = []

    for f in required:
        display = field_label(f)
        if f in values:
            lines.append(f"  [x] {display}: {values[f]}")
        else:
            lines.append(f"  [ ] {display}")

    return "These are the values I have registered:\n\n" + "\n\n".join(lines)


def _scenario(state: WorkflowState) -> str:
    return state.get("stub_scenario", "happy_path")


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json_response(text: str) -> dict:
    """Extract a JSON object from an LLM response, stripping markdown fences."""
    text = text.strip()
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Async LLM helpers
# ---------------------------------------------------------------------------


async def _llm_classify_message(llm: Any, text: str, state: WorkflowState) -> bool:
    """Use the LLM to classify a user message as question vs workflow data.

    Returns True if the message is a question / off-workflow comment.
    """
    stage = state.get("stage", "START")
    expectation = _STAGE_EXPECTATIONS.get(stage, "workflow-related input")
    intent = state.get("intent", "")
    label = _INTENT_LABELS.get(intent, intent or "unknown")

    system = (
        "You are a banking workflow assistant. Classify the user's latest message.\n\n"
        f"Current workflow: {label}\n"
        f"Current stage: {stage}\n"
        f"Expected input at this stage: {expectation}\n\n"
        "Is the user providing workflow data (field values, confirmation, retry, "
        "cancel, or other action that progresses the workflow), or asking a question "
        "/ making a comment that does NOT progress the workflow?\n\n"
        'Return ONLY a JSON object: {"type": "question"} or {"type": "workflow_data"}'
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=text),
        ])
        parsed = _parse_json_response(response.content)
        return parsed.get("type") == "question"
    except Exception:
        logger.exception("LLM classify_message failed, defaulting to workflow_data")
    return False


async def _llm_classify_intent(llm: Any, text: str) -> dict[str, Any]:
    """Use the LLM to classify intent from free-text."""
    system = (
        "You are a banking assistant. Classify the user's intent.\n\n"
        "Possible intents:\n"
        "- ADD_BENEFICIARY: user wants to add a new beneficiary / payee\n"
        "- REQUEST_CREDIT_LIMIT_INCREASE: user wants to request a higher credit limit\n\n"
        "If the message doesn't clearly match either intent, return UNKNOWN.\n\n"
        'Return ONLY a JSON object, e.g.: {"intent": "REQUEST_CREDIT_LIMIT_INCREASE"}'
    )
    try:
        response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=text)])
        parsed = _parse_json_response(response.content)
        if "intent" in parsed:
            return parsed
    except Exception:
        logger.exception("LLM classify_intent failed, falling back to stub")
    return {}


async def _llm_extract_fields(
    llm: Any,
    latest_text: str,
    required_fields: list[str],
    field_values: dict[str, str],
) -> dict[str, str]:
    """Extract field values from the user's latest message only.

    Previous values are supplied via *field_values* so the LLM knows what
    has already been collected without re-reading the full conversation
    (which caused it to favour old values over update requests).
    """
    known = json.dumps(field_values) if field_values else "{}"
    fields_list = ", ".join(required_fields)
    system = (
        "You are extracting structured field values from a user message in a "
        "banking workflow.\n\n"
        f"Required fields: [{fields_list}]\n"
        f"Already collected: {known}\n\n"
        "Focus ONLY on the user's message below. Extract any field values the "
        "user is providing, updating, or correcting. The user may use natural "
        "language — they do not need to write key=value.\n\n"
        "If the user requests to CHANGE or UPDATE a field that is already "
        "collected, return the NEW value — do NOT echo back the old one.\n\n"
        "CRITICAL: You MUST use EXACTLY the field names listed above as JSON "
        "keys. Do NOT invent your own key names or use synonyms. Map the "
        "user's natural language to the exact required field name. For example, "
        "if the required field is 'sort_code', use 'sort_code' as the key "
        "even if the user says 'sort code', 'branch code', or 'routing'.\n\n"
        "Return ONLY a JSON object with field names as keys and extracted values "
        "as strings. If the message contains no field values, return {}."
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=latest_text),
        ])
        parsed = _parse_json_response(response.content)
        return {k: str(v) for k, v in parsed.items() if k in required_fields}
    except Exception:
        logger.exception("LLM extract_fields failed, falling back to stub")
    return {}


async def _llm_generate_response(llm: Any, state: WorkflowState) -> str:
    """Generate a user-facing response: answer questions or guide the workflow."""
    intent = state.get("intent", "")
    label = _INTENT_LABELS.get(intent, intent or "Unknown")
    stage = state.get("stage", "")
    field_values = state.get("field_values", {})
    missing = state.get("missing_fields", [])
    email_candidates = state.get("email_candidates", [])
    email_to = state.get("email_to", "")
    email_subject = state.get("email_subject", "")
    email_body = state.get("email_body", "")
    last_error = state.get("last_error")

    context_parts = [f"Workflow: {label}", f"Stage: {stage}"]
    if field_values:
        labelled_values = {field_label(k): v for k, v in field_values.items()}
        context_parts.append(f"Collected fields: {json.dumps(labelled_values)}")
    if missing:
        labelled_missing = [f'"{field_label(f)}"' for f in missing]
        context_parts.append(
            f"Missing fields (use EXACTLY these labels): {', '.join(labelled_missing)}"
        )
    if email_candidates:
        context_parts.append(f"Email candidates: {', '.join(email_candidates)}")
    if email_to:
        context_parts.append(f"Email recipient: {email_to}")
    if email_subject:
        context_parts.append(f"Email subject: {email_subject}")
    if email_body:
        context_parts.append(f"Email body: {email_body}")
    if last_error:
        context_parts.append(
            f"Error: {last_error.get('source', '?')}: {last_error.get('detail', '?')}"
        )

    is_question = state.get("is_question", False)
    if is_question:
        role_instruction = (
            "The user asked a question or made a comment. Answer it based on the "
            "workflow context below. Do not invent information."
        )
    else:
        role_instruction = _STAGE_INSTRUCTIONS.get(
            stage,
            "Guide the user through the current workflow step.",
        )

    system = (
        "You are a friendly banking assistant.\n"
        "You can ONLY help with two operations: adding a new beneficiary / payee "
        "and requesting a credit limit increase. You have NO other capabilities. NEVER offer to "
        "help with operations you cannot perform.\n\n"
        + role_instruction
        + "\n\nCurrent state:\n"
        + "\n".join(f"- {p}" for p in context_parts)
        + "\n\nKeep your response concise and natural."
    )
    clean_msgs = [
        m for m in state.get("messages", [])
        if not (hasattr(m, "name") and m.name == "step_notification")
    ]
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system),
            *clean_msgs,
        ])
        return response.content
    except Exception:
        logger.exception("LLM generate_response failed, falling back to stub")
    return _format_status_preamble(state)


# ---------------------------------------------------------------------------
# Nodes (non-LLM, module-level)
# ---------------------------------------------------------------------------


def get_template_node(state: WorkflowState) -> dict:
    intent = state.get("intent", "")
    old_stage = state.get("stage", "START")
    result = get_template(intent, scenario=_scenario(state))

    if "error" in result:
        return {
            "stage": "ERROR",
            "last_error": {"source": "get_template", "detail": result["error"]},
            "retry_node": "get_template",
        }

    new_stage = "ASK_FOR_FIELDS"
    return {
        "stage": new_stage,
        "template_id": result["template_id"],
        "required_fields": result["required_fields"],
        "template_body": result["template_body"],
        "last_error": None,
        "messages": _step_entry_messages(old_stage, new_stage),
    }


def ask_for_fields_node(state: WorkflowState) -> dict:
    required = state.get("required_fields", [])
    values = state.get("field_values", {})
    old_stage = state.get("stage", "START")
    missing = [f for f in required if f not in values]

    if missing:
        new_stage = "EXTRACT_FIELDS"
        return {
            "stage": new_stage,
            "missing_fields": missing,
            "messages": _step_entry_messages(old_stage, new_stage),
        }

    new_stage = "LOOKUP_CUSTOMER_EMAIL"
    return {
        "stage": new_stage,
        "missing_fields": [],
        "messages": _step_entry_messages(old_stage, new_stage),
    }


def lookup_customer_email_node(state: WorkflowState) -> dict:
    old_stage = state.get("stage", "START")

    if state.get("email_to"):
        new_stage = "CUSTOMER_BACKGROUND_CHECK"
        return {
            "stage": new_stage,
            "last_error": None,
            "messages": _step_entry_messages(old_stage, new_stage),
        }

    customer_id = state.get("field_values", {}).get("customer_id")
    if not customer_id:
        new_stage = "ASK_FOR_FIELDS"
        return {
            "stage": new_stage,
            "missing_fields": ["customer_id"],
            "messages": _step_entry_messages(old_stage, new_stage),
        }

    result = lookup_customer_email(customer_id, scenario=_scenario(state))

    if "candidates" in result:
        new_stage = "EXTRACT_FIELDS"
        return {
            "stage": new_stage,
            "email_candidates": result["candidates"],
            "retry_node": "lookup_customer_email",
            "messages": _step_entry_messages(old_stage, new_stage),
        }

    if "error" in result:
        return {
            "stage": "ERROR",
            "last_error": {"source": "lookup_customer_email", "detail": result["error"]},
            "retry_node": "lookup_customer_email",
        }

    new_stage = "CUSTOMER_BACKGROUND_CHECK"
    return {
        "stage": new_stage,
        "email_to": result["email"],
        "last_error": None,
        "messages": _step_entry_messages(old_stage, new_stage),
    }


def confirm_node(state: WorkflowState) -> dict:
    old_stage = state.get("stage", "START")
    template_id = state.get("template_id", "")
    field_values = state.get("field_values", {})
    email_to = state.get("email_to", "")

    rendered = render_email(
        template_id, field_values, email_to,
        template_body=state.get("template_body", ""),
        scenario=_scenario(state),
    )

    subject = rendered["subject"]
    body = rendered["body"]

    preview = (
        f"{_format_status_preamble(state)}\n\n"
        f"Here is your email preview:\n\n"
        f"**To:** {email_to}\n"
        f"**Subject:** {subject}\n\n"
        f"{body}\n\n"
        f"Reply **SEND** to confirm, or describe any changes you'd like."
    )

    new_stage = "SEND_EMAIL"
    return {
        "stage": new_stage,
        "email_subject": subject,
        "email_body": body,
        "messages": _step_entry_messages(old_stage, new_stage) + [AIMessage(content=preview)],
    }


def send_email_node(state: WorkflowState) -> dict:
    email_to = state.get("email_to", "")
    subject = state.get("email_subject", "")
    body = state.get("email_body", "")

    result = send_email(email_to, subject, body, scenario=_scenario(state))

    if result["status"] == "failed":
        return {
            "stage": "ERROR",
            "last_error": {
                "source": "send_email",
                "detail": result.get("error", "Unknown error"),
            },
            "retry_node": "send_email",
        }

    return {
        "stage": "DONE",
        "last_error": None,
    }


def customer_background_check_submit_node(state: WorkflowState) -> dict:
    """Submit an async background check and transition to the waiting stage."""
    old_stage = state.get("stage", "START")
    customer_id = state.get("field_values", {}).get("customer_id", "")

    result = submit_background_check(customer_id, scenario=_scenario(state))
    request_id = result["request_id"]
    logger.info("Background check submitted: request_id=%s, customer=%s", request_id, customer_id)

    new_stage = "WAITING_FOR_BACKGROUND_CHECK"
    msg = (
        f"A mandatory background check has been submitted for customer "
        f"**{customer_id}** (request `{request_id}`). "
        f"Please wait — you will be notified automatically when the results are ready."
    )
    return {
        "stage": new_stage,
        "background_check_request_id": request_id,
        "messages": _step_entry_messages(old_stage, new_stage) + [AIMessage(content=msg)],
    }


def customer_background_check_resume_node(state: WorkflowState) -> dict:
    """Process a background-check result: advance to CONFIRM or terminate on denial."""
    old_stage = state.get("stage", "START")
    bg_result = state.get("background_check_result", {})
    request_id = state.get("background_check_request_id", "")
    status = bg_result.get("status", "unknown") if isinstance(bg_result, dict) else "unknown"
    logger.info("Background check resumed: request_id=%s, result=%s", request_id, bg_result)

    if status == "denied":
        msg = (
            f"Background check complete (request `{request_id}`): **denied**.\n\n"
            f"Unfortunately, the background check for this customer was denied "
            f"and this process cannot continue. "
            f"Please start a new chat if you would like to begin a new request."
        )
        return {
            "stage": "DENIED",
            "background_check_result": None,
            "messages": [AIMessage(content=msg)],
        }

    new_stage = "CONFIRM"
    msg = (
        f"Background check complete (request `{request_id}`): **{status}**. "
        f"Proceeding to email confirmation.\n\n"
    )
    return {
        "stage": new_stage,
        "background_check_result": None,
        "messages": [AIMessage(content=msg)] + _step_entry_messages(old_stage, new_stage),
    }


def done_node(state: WorkflowState) -> dict:
    return {"stage": "DONE"}


def error_node(state: WorkflowState) -> dict:
    return {"stage": "ERROR"}


# ---------------------------------------------------------------------------
# Routing (conditional-edge functions)
# ---------------------------------------------------------------------------


def _route_by_stage(state: WorkflowState) -> str:
    stage = state.get("stage", "START")

    # WAITING lockdown: takes precedence over everything, including is_question.
    if stage == "WAITING_FOR_BACKGROUND_CHECK":
        if state.get("background_check_result"):
            return "customer_background_check_resume"
        return END

    if stage == "DENIED":
        return END

    if state.get("is_question"):
        return "respond"

    if stage in ("START", "CLASSIFY_INTENT"):
        return "classify_intent"

    if stage == "EXTRACT_FIELDS":
        return "extract_fields"

    if stage == "SEND_EMAIL":
        latest = _get_latest_user_text(state)
        if latest and latest.strip().lower() in _CONFIRM_WORDS:
            return "send_email"
        return "extract_fields"

    if stage == "ERROR":
        latest = _get_latest_user_text(state)
        if latest and "retry" in latest.lower():
            return state.get("retry_node") or "classify_intent"
        if latest and "cancel" in latest.lower():
            return "done"
        return "respond"

    if stage == "DONE":
        return END

    return "classify_intent"


def _after_classify(state: WorkflowState) -> str:
    if state.get("intent") == "UNKNOWN":
        return "respond"
    return "get_template"


def _after_get_template(state: WorkflowState) -> str:
    if state.get("last_error"):
        return "error"
    return "extract_fields"


def _after_ask_for_fields(state: WorkflowState) -> str:
    if state.get("missing_fields"):
        return "respond"
    return "lookup_customer_email"


def _after_extract_fields(state: WorkflowState) -> str:
    if state.get("stage") == "SEND_EMAIL":
        return "confirm"
    return "ask_for_fields"


def _after_lookup_email(state: WorkflowState) -> str:
    if state.get("last_error"):
        return "error"
    stage = state.get("stage", "")
    if stage == "ASK_FOR_FIELDS":
        return "ask_for_fields"
    if stage == "EXTRACT_FIELDS":
        return "respond"
    return "customer_background_check_submit"


def _after_background_check_resume(state: WorkflowState) -> str:
    if state.get("stage") == "DENIED":
        return END
    return "confirm"


def _after_send_email(state: WorkflowState) -> str:
    if state.get("last_error"):
        return "error"
    return "done"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None, llm=None):
    """Compile the workflow state machine.

    Args:
        checkpointer: LangGraph checkpointer (e.g. AsyncCheckpointSaver).
        llm: Optional LangChain chat model for message classification,
             intent classification, field extraction, and conversational
             responses. When *None*, nodes fall back to deterministic stubs.
    """

    # -- LLM-powered nodes (closures capturing *llm*) ----------------------

    async def router_node(state: WorkflowState) -> dict:
        """Classify the user's message, then let the conditional edge route."""
        stage = state.get("stage", "START")
        if stage in ("START", "CLASSIFY_INTENT"):
            msgs = _step_entry_messages("", "START") if stage == "START" else []
            return {"is_question": False, "messages": msgs}

        if stage == "WAITING_FOR_BACKGROUND_CHECK":
            if state.get("background_check_result"):
                return {"is_question": False}
            msg = (
                "A mandatory background check is currently in progress. "
                "Please wait — you will be notified automatically when "
                "the results are ready. The workflow cannot proceed until "
                "this step is complete."
            )
            return {"is_question": False, "messages": [AIMessage(content=msg)]}

        if stage == "DENIED":
            msg = (
                "This process has been terminated because the background "
                "check was denied. No further actions can be taken in this "
                "conversation. Please start a new chat if you would like "
                "to begin a new request."
            )
            return {"is_question": False, "messages": [AIMessage(content=msg)]}

        latest = _get_latest_user_text(state)
        if llm and latest:
            is_q = await _llm_classify_message(llm, latest, state)
            return {"is_question": is_q}

        return {"is_question": False}

    async def classify_intent_node(state: WorkflowState) -> dict:
        text = _get_latest_user_text(state) or ""
        old_stage = state.get("stage", "START")

        if llm:
            parsed = await _llm_classify_intent(llm, text)
            intent = parsed.get("intent", "").upper()
            if intent in ("ADD_BENEFICIARY", "REQUEST_CREDIT_LIMIT_INCREASE"):
                new_stage = "GET_TEMPLATE"
                return {
                    "stage": new_stage,
                    "intent": intent,
                    "messages": _step_entry_messages(old_stage, new_stage),
                }
            if intent == "UNKNOWN":
                return {
                    "stage": "CLASSIFY_INTENT",
                    "intent": "UNKNOWN",
                }

        result = classify_intent(text, scenario=_scenario(state))
        if result["intent"] == "UNKNOWN":
            return {
                "stage": "CLASSIFY_INTENT",
                "intent": "UNKNOWN",
            }
        new_stage = "GET_TEMPLATE"
        return {
            "stage": new_stage,
            "intent": result["intent"],
            "messages": _step_entry_messages(old_stage, new_stage),
        }

    async def extract_fields_node(state: WorkflowState) -> dict:
        required = state.get("required_fields", [])
        current_values = state.get("field_values", {})
        updates: dict[str, Any] = {}

        candidates = state.get("email_candidates", [])
        if candidates:
            text = _get_latest_user_text(state) or ""
            normalized = text.strip().lower()
            for candidate in candidates:
                if candidate.lower() in normalized or normalized == candidate.lower():
                    updates["email_to"] = candidate
                    updates["email_candidates"] = []
                    break

        text = _get_latest_user_text(state) or ""
        if llm:
            new_fields = await _llm_extract_fields(llm, text, required, current_values)
        else:
            new_fields = extract_fields(text, required, scenario=_scenario(state))

        merged = {**current_values, **new_fields}
        if (
            "customer_id" in new_fields
            and new_fields["customer_id"] != current_values.get("customer_id")
        ):
            updates["email_to"] = ""

        updates["field_values"] = merged
        came_from_send = state.get("stage") == "SEND_EMAIL"
        values_changed = bool(new_fields) or "email_to" in updates
        if came_from_send and not values_changed:
            updates["stage"] = "SEND_EMAIL"
        else:
            updates["stage"] = "ASK_FOR_FIELDS"
        return updates

    async def respond_node(state: WorkflowState) -> dict:
        preamble = _format_status_preamble(state)
        if llm:
            response = await _llm_generate_response(llm, state)
            combined = f"{preamble}\n\n---\n\n{response}"
        else:
            combined = preamble
        return {"messages": [AIMessage(content=combined)]}

    # -- Build graph --------------------------------------------------------

    builder = StateGraph(WorkflowState)

    builder.add_node("router", router_node)
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("get_template", get_template_node)
    builder.add_node("ask_for_fields", ask_for_fields_node)
    builder.add_node("extract_fields", extract_fields_node)
    builder.add_node("lookup_customer_email", lookup_customer_email_node)
    builder.add_node("customer_background_check_submit", customer_background_check_submit_node)
    builder.add_node("customer_background_check_resume", customer_background_check_resume_node)
    builder.add_node("confirm", confirm_node)
    builder.add_node("send_email", send_email_node)
    builder.add_node("done", done_node)
    builder.add_node("error", error_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", _route_by_stage)

    builder.add_conditional_edges("classify_intent", _after_classify)
    builder.add_conditional_edges("get_template", _after_get_template)
    builder.add_conditional_edges("ask_for_fields", _after_ask_for_fields)
    builder.add_conditional_edges("extract_fields", _after_extract_fields)
    builder.add_conditional_edges("lookup_customer_email", _after_lookup_email)
    builder.add_edge("customer_background_check_submit", END)
    builder.add_conditional_edges("customer_background_check_resume", _after_background_check_resume)
    builder.add_edge("confirm", END)
    builder.add_conditional_edges("send_email", _after_send_email)
    builder.add_edge("done", "respond")
    builder.add_edge("error", "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=checkpointer)
