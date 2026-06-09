"""Common manifest-based pipeline infrastructure.

Provides the shared init → execute → collect → cleanup lifecycle used by both
:class:`CollectorTask` and :class:`EnricherTask` pipelines.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from SPARQLWrapper import SPARQLWrapper
import pandas as pd
from pydantic import BaseModel

from people_gator_common.filesystem import delete, exists, makedirs, open_file, ls
from people_gator_common.rate_limiter import RateLimiter
from people_gator_common.transport.sparql import execute_sparql_query, parse_sparql_results


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task protocol — every task model must provide a ``frame_name``
# ---------------------------------------------------------------------------


class TaskModel(BaseModel):
    """Minimal base for manifest task entries.

    Subclasses **must** override :attr:`frame_name` with a unique
    deterministic file-name for the partial result CSV of this task.
    """

    name: str
    command: str

    @property
    def frame_name(self) -> str:
        raise NotImplementedError


class ManifestModel(BaseModel):
    """Minimal base for manifests.

    Subclasses must declare ``tasks: list[SomeTaskSubclass]``.
    """

    output_file: str


# ---------------------------------------------------------------------------
# Completed-frame tracking (shared between collectors & enrichers)
# ---------------------------------------------------------------------------


def get_completed_frames(directory: str) -> set[str]:
    """Return the set of CSV filenames already present in *directory*."""
    completed = set(
        map(
            lambda x: x.split("/")[-1],
            filter(lambda x: x.endswith(".csv"), ls(directory)),
        )
    )
    logger.info("Found %d completed frames in %s: %s", len(completed), directory, completed)
    return completed


# ---------------------------------------------------------------------------
# Generic manifest helpers
# ---------------------------------------------------------------------------


def get_manifest_tasks[T: TaskModel](
    directory: str,
    manifest_type: type[ManifestModel],
    task_key: Callable[[T], str],
) -> dict[str, T]:
    """Read a manifest from *directory*/manifest.json and index tasks by key.

    *task_key* extracts the indexing key from each task (typically
    ``task.frame_name``).
    """
    with open_file(f"{directory}/manifest.json", "r", encoding="utf-8") as f:
        manifest = manifest_type.model_validate_json(f.read())
        tasks: list[T] = manifest.tasks  # type: ignore[attr-defined]
        return {task_key(task): task for task in tasks}


def get_pending_tasks[T: TaskModel](
    directory: str,
    manifest_type: type[ManifestModel],
    task_key: Callable[[T], str],
) -> list[T]:
    """Return only the tasks whose frame is not yet in *directory*."""
    all_tasks = get_manifest_tasks(directory, manifest_type, task_key)
    completed = get_completed_frames(directory)
    return [task for key, task in all_tasks.items() if key not in completed]


# ---------------------------------------------------------------------------
# Init — write manifest (shared logic)
# ---------------------------------------------------------------------------


def init_manifest(
    tmp_dir_path: str,
    manifest: ManifestModel,
    *,
    force: bool,
) -> None:
    """Create *tmp_dir_path*/manifest.json from *manifest*, respecting *force*."""
    manifest_path = os.path.join(tmp_dir_path, "manifest.json")
    if exists(manifest_path) and not force:
        raise FileExistsError(
            f"Manifest already exists at {manifest_path}. Use force=True to overwrite."
        )
    if exists(manifest_path) and force:
        delete(tmp_dir_path, recursive=True)
        logger.warning("Manifest already exists at %s, deleted due to force=True", manifest_path)

    makedirs(tmp_dir_path, exist_ok=True)
    with open_file(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Execute — run SPARQL queries for pending tasks and save CSVs
# ---------------------------------------------------------------------------


def execute_tasks[T: TaskModel](
    tmp_dir: str,
    manifest_type: type[ManifestModel],
    task_key: Callable[[T], str],
    client: SPARQLWrapper,
    rate_limiter: RateLimiter,
) -> None:
    """Execute every pending task in *tmp_dir* and write CSV frames."""
    pending = get_pending_tasks(tmp_dir, manifest_type, task_key)
    for task in pending:
        logger.info(
            "Executing task %s (command length=%d)",
            task.name,
            len(task.command),
        )
        try:
            json_result = execute_sparql_query(
                client=client,
                query=task.command,
                rate_limiter=rate_limiter,
            )
        except Exception:
            logger.exception("Failed to execute task %s", task.name)
            raise
        df = parse_sparql_results(json_result)
        output_path = os.path.join(tmp_dir, task_key(task))
        with open_file(output_path, "w", encoding="utf-8") as f:
            df.to_csv(f, index=False)
        logger.info("Completed task %s, saved results to %s", task.name, output_path)


# ---------------------------------------------------------------------------
# Collect — concatenate all frames into a single CSV
# ---------------------------------------------------------------------------


def collect_results(
    tmp_dir: str,
    output_file: str,
    manifest_type: type[ManifestModel],
    task_key: Callable[..., str],
    *,
    sort_by: str | None = None,
) -> None:
    """Concatenate all completed frame CSVs and write *output_file*.

    *sort_by* — if given, sort the final DataFrame by that column before
    writing.  When ``None`` the concatenated order is preserved.
    """
    manifest_path = os.path.join(tmp_dir, "manifest.json")
    if not exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    all_tasks = get_manifest_tasks(tmp_dir, manifest_type, task_key)
    completed = get_completed_frames(tmp_dir)
    pending = [task for key, task in all_tasks.items() if key not in completed]
    if pending:
        raise RuntimeError(
            f"Cannot collect results because there are still pending tasks: "
            f"{[t.name for t in pending]}"
        )

    dataframes: list[pd.DataFrame] = []
    for frame_name in sorted(completed):
        df = pd.read_csv(os.path.join(tmp_dir, frame_name))
        dataframes.append(df)

    if dataframes:
        combined = pd.concat(dataframes, ignore_index=True).drop_duplicates()
        if sort_by is not None:
            combined = combined.sort_values(by=sort_by)
        combined = combined.reset_index(drop=True)
        with open_file(output_file, "w", encoding="utf-8") as f:
            combined.to_csv(f, index=False)
        logger.info(
            "Collected %d total rows and saved to %s",
            len(combined),
            output_file,
        )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup(tmp_dir_path: str) -> None:
    """Recursively delete *tmp_dir_path*."""
    delete(tmp_dir_path, recursive=True)


# ---------------------------------------------------------------------------
# Full lifecycle runner
# ---------------------------------------------------------------------------


def run_pipeline[T: TaskModel](
    output_file: str,
    tmp_dir: str,
    manifest_type: type[ManifestModel],
    task_key: Callable[[T], str],
    init_fn: Callable[[], None],
    task_tmp_dir: str,
    *,
    sort_by: str | None = None,
    force: bool = False,
    client: SPARQLWrapper,
    rate_limiter: RateLimiter,
) -> None:
    """Run the full init → execute → collect → cleanup lifecycle.

    Parameters
    ----------
    output_file:
        Final merged CSV destination.
    tmp_dir:
        Root temporary directory.
    manifest_type:
        Pydantic model class for the manifest (``CollectorManifest`` or
        ``EnricherManifest``).
    task_key:
        Function that extracts the frame-file-name key from a task (usually
        ``lambda t: t.frame_name``).
    init_fn:
        Callable that writes the manifest to disk.  Must raise
        :class:`FileExistsError` when the manifest already exists and
        ``force`` is ``False``.
    task_tmp_dir:
        Directory where per-task CSV frames are written.
    sort_by:
        Column name to sort the concatenated result by, or ``None``.
    force:
        Overwrite existing outputs / manifests when ``True``.
    client:
        Configured SPARQL client.
    rate_limiter:
        Rate-limiting wrapper around the client.
    """
    if exists(output_file) and not force:
        logger.warning(
            "Output file %s already exists. Skipping because force=False.",
            output_file,
        )
        return
    try:
        init_fn()
    except FileExistsError as e:
        logger.warning("Initialization skipped: %s", str(e))
    execute_tasks(
        tmp_dir=task_tmp_dir,
        manifest_type=manifest_type,
        task_key=task_key,
        client=client,
        rate_limiter=rate_limiter,
    )
    collect_results(
        tmp_dir=task_tmp_dir,
        output_file=output_file,
        manifest_type=manifest_type,
        task_key=task_key,
        sort_by=sort_by,
    )
    cleanup(tmp_dir_path=task_tmp_dir)
