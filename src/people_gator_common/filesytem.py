from __future__ import annotations

import os
from typing import IO, Any, cast
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from people_gator_common.settings import PeopleGatorSettings

# Lazily-initialized filesystem instance
fs: AbstractFileSystem | None = None


def _setup_filesystem() -> AbstractFileSystem:
    """Create and return the configured fsspec filesystem instance."""
    global fs
    if fs is not None:
        return fs

    settings = PeopleGatorSettings()
    storage = settings.storage
    if storage == "local":
        fs = LocalFileSystem()
        return fs
    if storage == "s3":
        # import lazily to avoid hard dependency at import time
        from s3fs import S3FileSystem

        fs = S3FileSystem()
        return fs
    if storage == "lakefs":
        from lakefs_spec import LakeFSFileSystem

        fs = LakeFSFileSystem()
        return fs

    raise ValueError(f"Unsupported storage type: {storage}")


def get_fs() -> AbstractFileSystem:
    """Return the configured fsspec filesystem (initializes lazily)."""
    return _setup_filesystem()


def _to_str_path(path: str | os.PathLike) -> str:
    return str(path)


def exists(path: str | os.PathLike) -> bool:
    fs = get_fs()
    return fs.exists(_to_str_path(path))


def ls(path: str | os.PathLike) -> list[str]:
    fs = get_fs()
    return fs.ls(_to_str_path(path))

def glob(path: str | os.PathLike) -> list[str]:
    fs = get_fs()
    files = [str(p) for p in fs.glob(_to_str_path(path), detail=False)]
    return files

def ensure_parent_dir(path: str | os.PathLike) -> None:
    """Ensure parent directory for `path` exists when supported by the FS.

    This is a best-effort helper: local filesystems will create directories,
    object stores may no-op.
    """
    fs = get_fs()
    parent = os.path.dirname(_to_str_path(path))
    if not parent:
        return
    try:
        fs.makedirs(parent, exist_ok=True)
    except Exception:
        # Some remote filesystems don't implement makedirs; ignore failures
        return

def open_file(path: str | os.PathLike, mode: str = "rb", **kwargs) -> IO[Any]:
    """Open a file on the configured filesystem and return a file-like object.

    Additional keyword arguments (for example `encoding=`) are forwarded to
    the underlying filesystem `open` implementation.
    """
    fs = get_fs()
    return cast(IO[Any], fs.open(_to_str_path(path), mode=mode, **kwargs))


def read_text(path: str | os.PathLike, encoding: str = "utf-8") -> str:
    with open_file(path, mode="r", encoding=encoding) as fh:
        return fh.read()


def write_text(path: str | os.PathLike, data: str, encoding: str = "utf-8") -> None:
    ensure_parent_dir(path)
    with open_file(path, mode="w", encoding=encoding) as fh:
        fh.write(data)


def write_bytes(path: str | os.PathLike, data: bytes) -> None:
    ensure_parent_dir(path)
    with open_file(path, mode="wb") as fh:
        fh.write(data)

__all__ = [
    "get_fs",
    "exists",
    "ls",
    "ensure_parent_dir",
    "open_file",
    "read_text",
    "write_text",
    "write_bytes",
]
