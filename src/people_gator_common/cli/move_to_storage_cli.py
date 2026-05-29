
from __future__ import annotations

from pathlib import Path
import argparse
import os
import tqdm

from people_gator_common.filesystem import glob, write_bytes

def get_files_to_upload(
    local_files: list[Path],
    uploaded_files: set[str],
    src: Path,
    dest: str,
    diff_only: bool
) -> list[tuple[Path, str]]:
    """Determine which local files need to be uploaded based on existing files in storage."""
    files_to_upload: list[tuple[Path, str]] = []
    for path in local_files:
        if path.is_symlink() or path.is_dir():
            continue
        rel_path = path.relative_to(src)
        rel_path_str = str(rel_path)
        dest_path = os.path.join(dest, rel_path_str) if dest else rel_path_str
        if diff_only and dest_path in uploaded_files:
            continue
        files_to_upload.append((path, dest_path))
    return files_to_upload

def upload_directory(
    src: Path, 
    dest: str,
    diff_only: bool = True,
) -> int:
    """Upload a local directory to the configured storage.

    - `src` is a local filesystem Path pointing to a directory.
    - `dest` is a storage-relative destination directory (empty means storage root).
    - When `diff_only` is True, only files that have changed are uploaded.

    Returns the number of files uploaded (or that would be uploaded with dry-run).
    """
    if not src.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {src}")
    uploaded = 0
    src = src.resolve()
    rglobbed_files: list[Path] = list(tqdm.tqdm(src.rglob("*"), desc="Scanning files to upload"))
    print(f"Globing remote files in destination '{dest or '/'}' for diffing...")
    uploaded_files: set[str] = set(glob(os.path.join(dest, "**", "*"))) if diff_only else set()
    print(f"Found {len(uploaded_files)} existing files in destination for diffing")
    files_to_upload = get_files_to_upload(rglobbed_files, uploaded_files, src, dest, diff_only)
    print(f"Found {len(files_to_upload)} files to upload")
    for path, dest_path in tqdm.tqdm(files_to_upload, desc="Uploading files"):
        bytes_data = path.read_bytes()
        write_bytes(dest_path, bytes_data)
        uploaded += 1
    return uploaded


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a local directory to configured storage")
    p.add_argument("src", type=Path, help="Local source directory to upload")
    p.add_argument("dest", type=str, help="Destination path in storage (relative)")
    p.add_argument("--force", action="store_true", help="Force upload of all files, even if they exist in storage with the same content")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        count = upload_directory(args.src, dest=args.dest, diff_only=not args.force)
    except Exception as exc:
        print(f"Error: {exc}")
        return 2

    print(f"Uploaded {count} files to '{args.dest or '/'}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
