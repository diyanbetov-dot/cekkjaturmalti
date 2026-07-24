# Essentials/helpers/beta_sheet_logger.py - Google Sheets Beta Logger Helper
import os
import time
import logging
import json
from datetime import datetime

logger = logging.getLogger("beta_sheet_logger")

MAX_FIELD_LENGTH = 10000

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

def _post_payload(payload: dict) -> bool:
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
            resp = requests.post(url, json=payload_to_send, timeout=2.5, allow_redirects=True)
            return resp.status_code == 200
        except ImportError:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                return resp.status == 200
    except Exception as e:
        logger.warning("Google Sheets log POST failed (failing open): %s", e)
        return False

def create_log(
    log_id: str,
    input_text: str,
    initial_output: str,
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
