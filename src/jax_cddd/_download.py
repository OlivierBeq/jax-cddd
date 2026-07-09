"""Generic, dependency-free (stdlib-only) file download helper, shared by
``vocab.py`` and ``params.py`` to fetch release assets (GitHub Releases) on
demand. Kept dependency-free (no jax/numpy) so it stays importable from the
legacy TF1 environment too (see ``scripts/``).
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Union


def sha256sum(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_reporthook(label: str):
    """Returns a urlretrieve reporthook that only prints when the whole
    percentage changes (urlretrieve calls this once per chunk -- for
    fast/local transfers that's far too often to print every call)."""
    last_pct = [-1]

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        pct = downloaded * 100 // total_size
        if pct == last_pct[0]:
            return
        last_pct[0] = pct
        print(
            f"\rDownloading {label}: {pct:3d}% "
            f"({downloaded // (1 << 20)} / {total_size // (1 << 20)} MB)",
            end="",
            flush=True,
        )

    return reporthook


def download_file(
    url: str,
    dest: Union[str, Path],
    expected_sha256: Optional[str] = None,
    force: bool = False,
    label: Optional[str] = None,
) -> Path:
    """Download ``url`` to ``dest``, verifying its checksum if given.

    A no-op if ``dest`` already exists (unless ``force=True``), so this is safe
    to call unconditionally on every load. The download is written atomically
    (to a temp file in the same directory, then renamed into place), so a
    failed/interrupted download never leaves a corrupt file that a later call
    would mistake for a valid cached copy. On checksum mismatch, the bad
    download is removed (not left behind) and ``RuntimeError`` is raised.

    Args:
        url: The URL to fetch.
        dest: Destination path.
        expected_sha256: If given, the downloaded file's sha256 must match, or
            this raises.
        force: Re-download even if ``dest`` already exists.
        label: Human-readable name for the progress printout (defaults to
            ``dest``'s filename).

    Returns:
        ``dest`` (as a ``Path``), whether freshly downloaded or already present.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_make_reporthook(label or dest.name))
        print()  # newline after the progress line
        if expected_sha256 is not None:
            actual_sha256 = sha256sum(tmp_path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"Downloaded file checksum mismatch for {url}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
        tmp_path.replace(dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return dest
