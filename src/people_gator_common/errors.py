from __future__ import annotations
from collections.abc import Iterator
from http.client import IncompleteRead, RemoteDisconnected
import socket
from urllib.error import HTTPError, URLError


def iter_error_chain(error: Exception) -> Iterator[BaseException]:
    current: BaseException | None = error
    seen_ids: set[int] = set()
    while current is not None and id(current) not in seen_ids:
        seen_ids.add(id(current))
        yield current
        current = current.__cause__ if current.__cause__ is not None else current.__context__


def is_too_many_requests_error(error: Exception) -> bool:
    for chained_error in iter_error_chain(error):
        message = str(chained_error).lower()
        if "429" in message or "too many requests" in message:
            return True
    return False


def is_connection_reset_error(error: Exception) -> bool:
    for chained_error in iter_error_chain(error):
        if isinstance(chained_error, ConnectionResetError):
            return True

        if isinstance(chained_error, URLError):
            reason = chained_error.reason
            if isinstance(reason, ConnectionResetError):
                return True
            errno = getattr(reason, "errno", None)
            if errno == 104:
                return True

        message = str(chained_error).lower()
        if "connection reset by peer" in message or "errno 104" in message:
            return True

    return False


def is_retryable_http_error(error: Exception) -> bool:
    for chained_error in iter_error_chain(error):
        if isinstance(chained_error, HTTPError):
            return chained_error.code in {500, 502, 503, 504}

        message = str(chained_error).lower()
        if (
            "http error 500" in message
            or "http error 502" in message
            or "http error 503" in message
            or "http error 504" in message
        ):
            return True

    return False


def is_retryable_transport_error(error: Exception) -> bool:
    for chained_error in iter_error_chain(error):
        if isinstance(chained_error, (IncompleteRead, RemoteDisconnected, TimeoutError, socket.timeout)):
            return True

        if isinstance(chained_error, URLError) and isinstance(chained_error.reason, TimeoutError | socket.timeout):
            return True

        message = str(chained_error).lower()
        if (
            "incompleteread" in message
            or "remote end closed connection without response" in message
            or "timed out" in message
        ):
            return True

    return False

def is_retryable_error(error: Exception) -> bool:
    return (
        is_too_many_requests_error(error)
        or is_connection_reset_error(error)
        or is_retryable_http_error(error)
        or is_retryable_transport_error(error)
    )