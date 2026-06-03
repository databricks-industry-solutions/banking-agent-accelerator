"""Deterministic stub tools for the banking workflow state machine.

Each function accepts a ``scenario`` kwarg (default ``"happy_path"``) that
controls success/failure branches so the full graph can be exercised without
any real backend.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any


# ---------------------------------------------------------------------------
# classify_intent
# ---------------------------------------------------------------------------

def classify_intent(text: str, *, scenario: str = "happy_path") -> dict[str, Any]:
    """Classify user intent from free-text via substring heuristic."""
    if scenario == "unknown_intent":
        return {"intent": "UNKNOWN", "confidence": 0.0}

    lower = text.lower()
    if "beneficiary" in lower or "payee" in lower:
        intent = "ADD_BENEFICIARY"
    elif "credit" in lower or "limit" in lower:
        intent = "REQUEST_CREDIT_LIMIT_INCREASE"
    else:
        intent = "UNKNOWN"

    confidence = 0.51 if scenario == "low_confidence" else 0.97
    return {"intent": intent, "confidence": confidence}


# ---------------------------------------------------------------------------
# get_template
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, dict[str, Any]] = {
    "ADD_BENEFICIARY": {
        "template_id": "tmpl-add-beneficiary-v1",
        "required_fields": [
            "customer_id",
            "beneficiary_name",
            "beneficiary_account",
            "sort_code",
        ],
        "template_body": (
            "Dear {{customer_id}},\n\n"
            "We have added {{beneficiary_name}} (account "
            "{{beneficiary_account}}, sort code {{sort_code}}) as a payee "
            "on your account."
        ),
    },
    "REQUEST_CREDIT_LIMIT_INCREASE": {
        "template_id": "tmpl-credit-limit-increase-v1",
        "required_fields": [
            "customer_id",
            "card_id",
            "amount",
            "currency",
            "reason",
        ],
        "template_body": (
            "Dear {{customer_id}},\n\n"
            "We have received your request to increase the credit limit on "
            "card {{card_id}} to {{amount}} {{currency}}. "
            "Reason: {{reason}}."
        ),
    },
}


_FIELD_LABELS: dict[str, str] = {
    "customer_id":         "Customer ID",
    "beneficiary_name":    "Beneficiary name",
    "beneficiary_account": "Beneficiary account number",
    "sort_code":           "Sort code",
    "card_id":             "Card ID",
    "amount":              "Requested credit limit",
    "currency":            "Currency",
    "reason":              "Reason for request",
}


def field_label(name: str) -> str:
    """Return the user-facing display label for a raw field name."""
    return _FIELD_LABELS.get(name, name)


def get_template(intent: str, *, scenario: str = "happy_path") -> dict[str, Any]:
    """Return the email template for the given intent."""
    if scenario == "template_not_found":
        return {"error": f"No template found for intent '{intent}'."}

    tmpl = _TEMPLATES.get(intent)
    if tmpl is None:
        return {"error": f"No template found for intent '{intent}'."}
    return dict(tmpl)


# ---------------------------------------------------------------------------
# extract_fields
# ---------------------------------------------------------------------------

_KV_RE = re.compile(r"(\w+)\s*=\s*(\S+)")


def extract_fields(
    text: str,
    required_fields: list[str],
    *,
    scenario: str = "happy_path",
) -> dict[str, str]:
    """Parse key=value pairs (or a JSON object) from user text.

    Only returns values whose keys are in *required_fields*.
    """
    if scenario == "missing_fields":
        return {}

    extracted: dict[str, str] = {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            extracted.update({k: str(v) for k, v in parsed.items()})
    except (json.JSONDecodeError, TypeError):
        pass

    for key, value in _KV_RE.findall(text):
        extracted[key] = value

    return {k: v for k, v in extracted.items() if k in required_fields}


# ---------------------------------------------------------------------------
# lookup_customer_email
# ---------------------------------------------------------------------------

def lookup_customer_email(
    customer_id: str,
    *,
    scenario: str = "happy_path",
) -> dict[str, Any]:
    """Look up the customer's email address."""
    if scenario == "ambiguous_email":
        return {
            "candidates": [
                f"{customer_id}@example.com",
                f"{customer_id}.alt@example.com",
            ]
        }

    return {"email": f"{customer_id}@example.com"}


# ---------------------------------------------------------------------------
# render_email
# ---------------------------------------------------------------------------

_SUBJECT_LABELS: dict[str, str] = {
    "tmpl-add-beneficiary-v1": "New Payee Added",
    "tmpl-credit-limit-increase-v1": "Your Credit Limit Increase Request",
}

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def render_email(
    template_id: str,
    field_values: dict[str, str],
    email_to: str,
    *,
    template_body: str = "",
    scenario: str = "happy_path",
) -> dict[str, str]:
    """Render the email subject and body from template + field values."""
    subject = _SUBJECT_LABELS.get(template_id, f"Notification for {email_to}")

    if template_body:
        body = _PLACEHOLDER_RE.sub(
            lambda m: field_values.get(m.group(1), m.group(0)),
            template_body,
        )
    else:
        fields_summary = ", ".join(f"{k}={v}" for k, v in sorted(field_values.items()))
        body = f"Generated from template {template_id} with:\n{fields_summary}"

    return {"subject": subject, "body": body}


# ---------------------------------------------------------------------------
# submit_background_check
# ---------------------------------------------------------------------------

def submit_background_check(
    customer_id: str,
    *,
    scenario: str = "happy_path",
) -> dict[str, str]:
    """Submit an async background check for the customer.

    Returns a request_id that can later be used to correlate the result.
    """
    return {"request_id": f"bgc-{uuid.uuid4().hex[:12]}"}


# ---------------------------------------------------------------------------
# send_email
# ---------------------------------------------------------------------------

def send_email(
    to: str,
    subject: str,
    body: str,
    *,
    scenario: str = "happy_path",
) -> dict[str, str]:
    """Send the rendered email."""
    if scenario == "send_failure":
        return {"status": "failed", "error": "SMTP connection refused (stub)."}

    return {"status": "sent", "message_id": f"stub-{uuid.uuid4().hex[:12]}"}
