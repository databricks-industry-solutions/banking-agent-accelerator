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
    if "statement" in lower:
        intent = "GENERATE_ACCOUNT_STATEMENT"
    elif "deposit" in lower:
        intent = "OPEN_DEPOSIT"
    else:
        intent = "UNKNOWN"

    confidence = 0.51 if scenario == "low_confidence" else 0.97
    return {"intent": intent, "confidence": confidence}


# ---------------------------------------------------------------------------
# get_template
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, dict[str, Any]] = {
    "GENERATE_ACCOUNT_STATEMENT": {
        "template_id": "tmpl-account-statement-v1",
        "required_fields": [
            "customer_id",
            "account_id",
            "period_start",
            "period_end",
        ],
        "template_body": (
            "Dear {{customer_id}},\n\n"
            "Please find attached your account statement for "
            "{{account_id}} covering {{period_start}} to {{period_end}}."
        ),
    },
    "OPEN_DEPOSIT": {
        "template_id": "tmpl-open-deposit-v1",
        "required_fields": [
            "customer_id",
            "amount",
            "currency",
            "term_months",
            "payout_account",
        ],
        "template_body": (
            "Dear {{customer_id}},\n\n"
            "We have opened a deposit of {{amount}} {{currency}} "
            "for {{term_months}} months. Payouts go to {{payout_account}}."
        ),
    },
}


_FIELD_LABELS: dict[str, str] = {
    "customer_id":    "Customer ID",
    "account_id":     "Account ID",
    "period_start":   "Statement start date",
    "period_end":     "Statement end date",
    "amount":         "Deposit amount",
    "currency":       "Currency",
    "term_months":    "Term (months)",
    "payout_account": "Payout account",
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
    "tmpl-account-statement-v1": "Your Account Statement",
    "tmpl-open-deposit-v1": "Your Deposit Confirmation",
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
