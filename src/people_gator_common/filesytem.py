from __future__ import annotations

import os
from typing import IO, Any, cast
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from people_gator_common.settings import PeopleGatorSettings

# Lazily-initialized filesystem instance
fs: AbstractFileSystem | None = None
prefix: str | None = None

def _setup_filesystem() -> AbstractFileSystem:
    """Create and return the configured fsspec filesystem instance."""
    global fs, prefix
    if fs is not None:
        return fs

    settings = PeopleGatorSettings()
    storage = settings.storage
    all_enabled = [
        ("local", storage.local.enabled),
        ("s3", storage.s3.enabled),
        ("lakefs", storage.lakefs.enabled),
    ]
    enabled_types = [name for name, enabled in all_enabled if enabled]
    if len(enabled_types) == 0:
        raise ValueError("No storage type enabled in settings")
    if len(enabled_types) > 1:
        raise ValueError(f"Multiple storage types enabled: {', '.join(enabled_types)}. Only one can be enabled at a time.")
    if storage.local.enabled:
        if not storage.local.root:
            raise ValueError("Local storage enabled but no root path specified")
        fs = LocalFileSystem()
        prefix = os.path.abspath(storage.local.root)
        return fs
    if storage.s3.enabled:
        # import lazily to avoid hard dependency at import time
        from s3fs import S3FileSystem
        fs = S3FileSystem(
            key=storage.s3.access_key_id,
            secret=storage.s3.secret_access_key,
            endpoint_url=storage.s3.endpoint_url,
        )
        prefix = storage.s3.bucket
        return fs
    if storage.lakefs.enabled:
        from lakefs_spec import LakeFSFileSystem
        fs = LakeFSFileSystem()
        return fs

    raise ValueError(f"Unsupported storage type: {storage}")


def get_fs() -> AbstractFileSystem:
    """Return the configured fsspec filesystem (initializes lazily)."""
    return _setup_filesystem()

def _realpath(path: str) -> str:
    return os.path.join(prefix, path) if prefix else path

def _relpath(path: str) -> str:
    if prefix and path.startswith(prefix):
        return os.path.relpath(path, prefix)
    return path

def exists(path: str) -> bool:
    fs = get_fs()
    return fs.exists(_realpath(path))

def delete(path: str) -> None:
    fs = get_fs()
    if fs.exists(_realpath(path)):
        fs.delete(_realpath(path))

def ls(path: str) -> list[str]:
    fs = get_fs()
    return [_relpath(p) for p in fs.ls(_realpath(path))]

def glob(path: str) -> list[str]:
    fs = get_fs()
    files = [_relpath(str(p)) for p in fs.glob(_realpath(path), detail=False)]
    return files

def makedirs(path: str, exist_ok: bool = True) -> None:
    fs = get_fs()
    try:
        fs.makedirs(_realpath(path), exist_ok=exist_ok)
    except Exception:
        # Some remote filesystems don't implement makedirs; ignore failures
        return

def ensure_parent_dir(path: str) -> None:
    """Ensure parent directory for `path` exists when supported by the FS.

    This is a best-effort helper: local filesystems will create directories,
    object stores may no-op.
    """
    parent = os.path.dirname(path)
    if not parent:
        return
    makedirs(parent, exist_ok=True)

def open_file(path: str, mode: str = "rb", **kwargs) -> IO[Any]:
    """Open a file on the configured filesystem and return a file-like object.

    Additional keyword arguments (for example `encoding=`) are forwarded to
    the underlying filesystem `open` implementation.
    """
    fs = get_fs()
    return cast(IO[Any], fs.open(_realpath(path), mode=mode, **kwargs))

def read_text(path: str, encoding: str = "utf-8") -> str:
    with open_file(path, mode="r", encoding=encoding) as fh:
        return fh.read()

def read_bytes(path: str) -> bytes:
    with open_file(path, mode="rb") as fh:
        return fh.read()

def write_text(path: str, data: str, encoding: str = "utf-8") -> None:
    ensure_parent_dir(path)
    with open_file(path, mode="w", encoding=encoding) as fh:
        fh.write(data)


def write_bytes(path: str, data: bytes) -> None:
    ensure_parent_dir(path)
    with open_file(path, mode="wb") as fh:
        fh.write(data)

__all__ = [
    "get_fs",
    "exists",
    "ls",
    "glob",
    "makedirs",
    "ensure_parent_dir",
    "open_file",
    "read_text",
    "read_bytes",
    "write_text",
    "write_bytes",
]
