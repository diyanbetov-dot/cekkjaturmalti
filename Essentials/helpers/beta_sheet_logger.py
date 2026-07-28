# Essentials/helpers/beta_sheet_logger.py - Google Sheets Beta Logger Helper
import os
import time
import logging
import json
from datetime import datetime

logger = logging.getLogger("beta_sheet_logger")

MAX_FIELD_LENGTH = 10000
MAX_FEEDBACK_MESSAGE_LENGTH = 10000
MAX_SCREENSHOT_DATA_URL_LENGTH = 3_000_000

def is_logging_enabled() -> bool:
    flag = os.getenv("SPELLCHECK_BETA_LOGGING", "false").strip().lower()
    return flag in ("true", "1", "yes")

def get_log_config() -> tuple[str, str]:
    url = os.getenv("SPELLCHECK_LOG_URL", "").strip()
    secret = os.getenv("SPELLCHECK_LOG_SECRET", "").strip()
    return url, secret

def format_timestamp(dt: datetime = None) -> str:
    if dt is None:
        dt = datetime.now()
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    day = f"{dt.day:02d}"
    month = month_names[dt.month - 1]
    year = dt.year
    hms = dt.strftime("%H:%M:%S")
    return f"{day}/{month}/{year} {hms}"

def build_initial_notes(tokens: list[dict]) -> str:
    if not isinstance(tokens, list):
        return ""

    note_lines = []
    unrecognized_words = []

    for token in tokens:
        if not isinstance(token, dict):
            continue

        text = token.get("corrected") or token.get("text") or ""
        choices = token.get("choices")

        if choices and isinstance(choices, list) and len(choices) > 0:
            extracted_choices = []
            for c in choices:
                if isinstance(c, str):
                    extracted_choices.append(c)
                elif isinstance(c, dict):
                    w = (
                        c.get("word")
                        or c.get("corrected")
                        or c.get("candidate")
                        or c.get("surface")
                    )
                    if w:
                        extracted_choices.append(w)
            if extracted_choices:
                # Deduplicate preserving order
                unique_choices = list(dict.fromkeys(extracted_choices))
                note_lines.append(f"{text} - suggestions: {', '.join(unique_choices)}.")

        if token.get("unrecognized"):
            orig_or_corr = token.get("corrected") or token.get("text") or ""
            if orig_or_corr and orig_or_corr not in unrecognized_words:
                unrecognized_words.append(orig_or_corr)

    if unrecognized_words:
        note_lines.append(f"Unrecognized words: {', '.join(unrecognized_words)}.")

    return "\n".join(note_lines) if note_lines else ""

def _post_payload(payload: dict, timeout: float = 2.5) -> bool:
    if not is_logging_enabled():
        return False

    url, secret = get_log_config()
    if not url or not secret:
        logger.warning("Beta Sheets logging enabled but SPELLCHECK_LOG_URL or SPELLCHECK_LOG_SECRET is missing.")
        return False

    payload_to_send = dict(payload)
    payload_to_send["secret"] = secret

    data_bytes = json.dumps(payload_to_send, ensure_ascii=False).encode("utf-8")

    # Attempt to send via requests if available, fallback to urllib.request
    try:
        try:
            import requests
            resp = requests.post(
                url,
                json=payload_to_send,
                timeout=timeout,
                allow_redirects=True,
            )
            if resp.status_code != 200:
                return False
            try:
                response_payload = resp.json()
            except (TypeError, ValueError):
                return True
            if (
                isinstance(response_payload, dict)
                and response_payload.get("ok") is False
            ):
                logger.warning(
                    "Google Apps Script rejected payload: %s",
                    response_payload.get("error", "unknown error"),
                )
                return False
            return True
        except ImportError:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return False
                try:
                    response_payload = json.loads(resp.read().decode("utf-8"))
                except (AttributeError, TypeError, ValueError, UnicodeDecodeError):
                    return True
                if (
                    isinstance(response_payload, dict)
                    and response_payload.get("ok") is False
                ):
                    logger.warning(
                        "Google Apps Script rejected payload: %s",
                        response_payload.get("error", "unknown error"),
                    )
                    return False
                return True
    except Exception as e:
        logger.warning("Google Sheets log POST failed (failing open): %s", e)
        return False

def create_log(
    log_id: str,
    input_text: str,
    initial_output: str,
    notes: str = None,
    final_output: str = None,
    timestamp: str = None
) -> bool:
    if not is_logging_enabled():
        return False

    if not log_id or not isinstance(log_id, str):
        logger.warning("create_log called with invalid log_id.")
        return False

    if final_output is None:
        final_output = initial_output

    if timestamp is None:
        timestamp = format_timestamp()

    payload = {
        "action": "create_log",
        "log_id": str(log_id)[:100],
        "timestamp": str(timestamp)[:100],
        "input": str(input_text or "")[:MAX_FIELD_LENGTH],
        "initial_output": str(initial_output or "")[:MAX_FIELD_LENGTH],
        "notes": str(notes or "")[:MAX_FIELD_LENGTH],
        "final_output": str(final_output or "")[:MAX_FIELD_LENGTH],
    }

    return _post_payload(payload)

def update_choice(
    log_id: str,
    event_id: str,
    token: str,
    suggestions: list[str],
    chosen: str,
    final_output: str
) -> bool:
    if not is_logging_enabled():
        return False

    if not log_id or not isinstance(log_id, str):
        logger.warning("update_choice called with missing or invalid log_id.")
        return False

    clean_suggestions = []
    if isinstance(suggestions, list):
        clean_suggestions = [str(s)[:200] for s in suggestions if s]

    payload = {
        "action": "update_choice",
        "log_id": str(log_id)[:100],
        "event_id": str(event_id or "")[:100],
        "token": str(token or "")[:200],
        "suggestions": clean_suggestions,
        "chosen": str(chosen or "")[:200],
        "final_output": str(final_output or "")[:MAX_FIELD_LENGTH],
    }

    return _post_payload(payload)


def submit_feedback(
    *,
    email: str,
    subject: str,
    message: str,
    screenshot_data_url: str = "",
    screenshot_filename: str = "cekkjatur-report.jpg",
    language: str = "mt",
    page_url: str = "",
    user_agent: str = "",
    reported_word: str = "",
    log_id: str = "",
) -> bool:
    """Send a feedback report through the configured Google Apps Script endpoint."""
    if not is_logging_enabled():
        return False

    payload = {
        "action": "feedback_report",
        "timestamp": format_timestamp(),
        "email": str(email or "")[:254],
        "subject": str(subject or "")[:200],
        "message": str(message or "")[:MAX_FEEDBACK_MESSAGE_LENGTH],
        "screenshot_data_url": str(screenshot_data_url or "")[
            :MAX_SCREENSHOT_DATA_URL_LENGTH
        ],
        "screenshot_filename": str(screenshot_filename or "")[:150],
        "language": str(language or "")[:10],
        "page_url": str(page_url or "")[:1000],
        "user_agent": str(user_agent or "")[:500],
        "reported_word": str(reported_word or "")[:200],
        "log_id": str(log_id or "")[:100],
    }

    # A screenshot upload is larger than the ordinary spellcheck log payload.
    return _post_payload(payload, timeout=12.0)
