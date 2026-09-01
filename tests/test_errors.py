from __future__ import annotations

from http.client import IncompleteRead, RemoteDisconnected
import socket
from urllib.error import URLError

from people_gator_common.errors import is_retryable_error


def test_is_retryable_error_for_incomplete_read() -> None:
    error = IncompleteRead(b"partial")
    assert is_retryable_error(error) is True


def test_is_retryable_error_for_remote_disconnected() -> None:
    error = RemoteDisconnected("Remote end closed connection without response")
    assert is_retryable_error(error) is True


def test_is_retryable_error_for_timeout_url_error() -> None:
    error = URLError(socket.timeout("timed out"))
    assert is_retryable_error(error) is True


def test_is_retryable_error_for_wrapped_incomplete_read() -> None:
    wrapped_error: RuntimeError
    try:
        try:
            raise IncompleteRead(b"partial")
        except IncompleteRead as incomplete_read_error:
            raise RuntimeError("wrapper") from incomplete_read_error
    except RuntimeError as runtime_error:
        wrapped_error = runtime_error

    assert is_retryable_error(wrapped_error) is True
