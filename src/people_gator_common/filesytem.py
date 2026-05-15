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
    if storage.local.enabled:
        fs = LocalFileSystem()
        return fs
    if storage.s3.enabled:
        # import lazily to avoid hard dependency at import time
        from s3fs import S3FileSystem
        fs = S3FileSystem(
            key=storage.s3.access_key_id,
            secret=storage.s3.secret_access_key,
            endpoint_url=storage.s3.endpoint_url,
        )
        return fs
    if storage.lakefs.enabled:
        from lakefs_spec import LakeFSFileSystem
        fs = LakeFSFileSystem()
        return fs

    raise ValueError(f"Unsupported storage type: {storage}")


def get_fs() -> AbstractFileSystem:
    """Return the configured fsspec filesystem (initializes lazily)."""
    return _setup_filesystem()


def _to_str_path(path: str) -> str:
    return str(path)

def exists(path: str) -> bool:
    fs = get_fs()
    return fs.exists(_to_str_path(path))

def delete(path: str) -> None:
    fs = get_fs()
    if fs.exists(_to_str_path(path)):
        fs.delete(_to_str_path(path))

def ls(path: str) -> list[str]:
    fs = get_fs()
    return fs.ls(_to_str_path(path))

def glob(path: str) -> list[str]:
    fs = get_fs()
    files = [str(p) for p in fs.glob(_to_str_path(path), detail=False)]
    return files

def makedirs(path: str, exist_ok: bool = True) -> None:
    fs = get_fs()
    try:
        fs.makedirs(_to_str_path(path), exist_ok=exist_ok)
    except Exception:
        # Some remote filesystems don't implement makedirs; ignore failures
        return

def ensure_parent_dir(path: str) -> None:
    """Ensure parent directory for `path` exists when supported by the FS.

    This is a best-effort helper: local filesystems will create directories,
    object stores may no-op.
    """
    parent = os.path.dirname(_to_str_path(path))
    if not parent:
        return
    makedirs(parent, exist_ok=True)
    

def open_file(path: str, mode: str = "rb", **kwargs) -> IO[Any]:
    """Open a file on the configured filesystem and return a file-like object.

    Additional keyword arguments (for example `encoding=`) are forwarded to
    the underlying filesystem `open` implementation.
    """
    fs = get_fs()
    delete(path)
    return cast(IO[Any], fs.open(_to_str_path(path), mode=mode, **kwargs))


def read_text(path: str, encoding: str = "utf-8") -> str:
    with open_file(path, mode="r", encoding=encoding) as fh:
        return fh.read()


def write_text(path: str, data: str, encoding: str = "utf-8") -> None:
    ensure_parent_dir(path)
    delete(path)
    with open_file(path, mode="w", encoding=encoding) as fh:
        fh.write(data)


def write_bytes(path: str, data: bytes) -> None:
    ensure_parent_dir(path)
    delete(path)
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
    "write_text",
    "write_bytes",
]
