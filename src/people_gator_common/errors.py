from __future__ import annotations
from urllib.error import HTTPError, URLError


def is_too_many_requests_error(error: Exception) -> bool:
    message = str(error).lower()
    return "429" in message or "too many requests" in message


def is_connection_reset_error(error: Exception) -> bool:
    if isinstance(error, ConnectionResetError):
        return True

    if isinstance(error, URLError):
        reason = error.reason
        if isinstance(reason, ConnectionResetError):
            return True
        errno = getattr(reason, "errno", None)
        if errno == 104:
            return True

    message = str(error).lower()
    return "connection reset by peer" in message or "errno 104" in message


def is_retryable_http_error(error: Exception) -> bool:
    if isinstance(error, HTTPError):
        return error.code in {500, 502, 503, 504}

    message = str(error).lower()
    return (
        "http error 500" in message
        or "http error 502" in message
        or "http error 503" in message
        or "http error 504" in message
    )

def is_retryable_error(error: Exception) -> bool:
    return (
        is_too_many_requests_error(error)
        or is_connection_reset_error(error)
        or is_retryable_http_error(error)
    )