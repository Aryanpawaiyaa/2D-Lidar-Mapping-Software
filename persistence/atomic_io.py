"""Crash-safe atomic file I/O operations."""

import os
import tempfile


def atomic_write_bytes(filepath: str, data: bytes) -> None:
    """Write bytes to disk atomically using temporary file and atomic rename."""
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)

    # Create temporary file in the same directory (crucial for same-filesystem atomic rename)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_atomic_")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def atomic_write_text(filepath: str, text: str, encoding: str = "utf-8") -> None:
    """Write text to disk atomically."""
    atomic_write_bytes(filepath, text.encode(encoding))

