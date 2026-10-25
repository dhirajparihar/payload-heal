"""Thread-safe and atomic file-locking reader/writer for the payload heal lockfile.

Lockfile Schema:
{
    "version": "1.0",
    "models": {
        "ModelName": {
            "field_name": {
                "source_path": "a.b.c",
                "healed_by": "heuristic|ai",
                "updated_at": "ISO-8601"
            }
        }
    }
}
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterator

# Try importing filelock for cross-platform inter-process locking
try:
    from filelock import FileLock  # type: ignore[import-untyped]
except ImportError:
    FileLock = None  # Fallback to OS-specific primitives or threading lock


DEFAULT_LOCKFILE_PATH = Path(".payload_heal_lock.json")


class LockFileManager:
    """Manages reading, writing, and atomic updates to the `.payload_heal_lock.json` cache.

    Guarantees:
    - Intra-process thread safety via `threading.RLock`.
    - Inter-process synchronization via OS / file locks.
    - Zero-corruption atomicity via tempfile write + `os.replace`.
    """

    def __init__(self, lockfile_path: str | Path = DEFAULT_LOCKFILE_PATH) -> None:
        self.lockfile_path = Path(lockfile_path).resolve()
        self._thread_lock = threading.RLock()
        self._lock_file_path = self.lockfile_path.with_name(f"{self.lockfile_path.name}.lock")

    @contextlib.contextmanager
    def _acquire_file_lock(self) -> Iterator[None]:
        """Context manager providing cross-process file locking."""
        if FileLock is not None:
            # Ensure parent directory exists for lockfile
            self._lock_file_path.parent.mkdir(parents=True, exist_ok=True)
            flock = FileLock(str(self._lock_file_path), timeout=10.0)
            with flock:
                yield
            return

        # Fallback to standard library OS-level locks if filelock is not installed
        lock_fd: int | None = None
        self._lock_file_path.parent.mkdir(parents=True, exist_ok=True)

        if os.name == "nt":
            import msvcrt

            try:
                lock_fd = os.open(str(self._lock_file_path), os.O_CREAT | os.O_RDWR)
                msvcrt.locking(lock_fd, msvcrt.LK_LOCK, 1)
                yield
            finally:
                if lock_fd is not None:
                    try:
                        msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                    os.close(lock_fd)
        else:
            import fcntl

            try:
                lock_fd = os.open(str(self._lock_file_path), os.O_CREAT | os.O_RDWR)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                yield
            finally:
                if lock_fd is not None:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                    os.close(lock_fd)

    def _read_raw(self) -> dict[str, Any]:
        """Reads and parses the lockfile JSON. Returns default structure on missing/corrupt."""
        if not self.lockfile_path.exists():
            return {"version": "1.0", "models": {}}

        try:
            content = self.lockfile_path.read_text(encoding="utf-8")
            if not content.strip():
                return {"version": "1.0", "models": {}}
            data = json.loads(content)
            if not isinstance(data, dict):
                return {"version": "1.0", "models": {}}
            data.setdefault("version", "1.0")
            if not isinstance(data.get("models"), dict):
                data["models"] = {}
            return data
        except Exception:
            return {"version": "1.0", "models": {}}

    def _write_raw(self, data: dict[str, Any]) -> None:
        """Atomically writes data to the lockfile path via tempfile + os.replace."""
        parent = self.lockfile_path.parent
        parent.mkdir(parents=True, exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            delete=False,
            prefix=f".{self.lockfile_path.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_file.name)
        try:
            json.dump(data, temp_file, indent=2, ensure_ascii=False)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            os.replace(temp_path, self.lockfile_path)
        except Exception:
            temp_file.close()
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise

    def get_mapping(self, model_name: str) -> dict[str, str]:
        """Retrieves cached field -> source_path mappings for the specified model.

        Parameters:
            model_name: The name of the Pydantic model class.

        Returns:
            Dictionary mapping `field_name` to healed `source_path`.
            Returns `{}` if the model is not found in the lockfile.
        """
        with self._thread_lock:
            with self._acquire_file_lock():
                data = self._read_raw()
                models = data.get("models", {})
                model_data = models.get(model_name, {})
                if not isinstance(model_data, dict):
                    return {}

                mapping: dict[str, str] = {}
                for field_name, entry in model_data.items():
                    if isinstance(entry, dict) and "source_path" in entry:
                        mapping[field_name] = str(entry["source_path"])
                return mapping

    def get_field_info(self, model_name: str, field_name: str) -> dict[str, Any] | None:
        """Returns the full metadata dictionary for a specific healed field."""
        with self._thread_lock:
            with self._acquire_file_lock():
                data = self._read_raw()
                model_data = data.get("models", {}).get(model_name, {})
                if isinstance(model_data, dict):
                    entry = model_data.get(field_name)
                    if isinstance(entry, dict):
                        return dict(entry)
                return None

    def save_mapping(
        self,
        model_name: str,
        field_name: str,
        source_path: str,
        healed_by: str,
    ) -> None:
        """Thread-safe and atomic persistence of a healed field path mapping.

        Parameters:
            model_name: Name of the target Pydantic model.
            field_name: Target attribute field on the model.
            source_path: Traversal path in the upstream payload (e.g. `items[0].price`).
            healed_by: Strategy used to heal (`heuristic` or `ai`).
        """
        with self._thread_lock:
            with self._acquire_file_lock():
                data = self._read_raw()
                models = data.setdefault("models", {})
                model_dict = models.setdefault(model_name, {})

                now_iso = datetime.now(timezone.utc).isoformat()
                model_dict[field_name] = {
                    "source_path": source_path,
                    "healed_by": healed_by,
                    "updated_at": now_iso,
                }

                self._write_raw(data)

    def invalidate_field(self, model_name: str, field_name: str) -> bool:
        """Invalidates a previously healed field mapping if the schema drifts again.

        Parameters:
            model_name: Target Pydantic model name.
            field_name: The field to invalidate.

        Returns:
            True if the field existed and was removed, False otherwise.
        """
        with self._thread_lock:
            with self._acquire_file_lock():
                data = self._read_raw()
                models = data.get("models", {})
                if model_name not in models:
                    return False

                model_dict = models[model_name]
                if not isinstance(model_dict, dict) or field_name not in model_dict:
                    return False

                del model_dict[field_name]

                # Prune empty model entries
                if not model_dict:
                    del models[model_name]

                self._write_raw(data)
                return True

    def remove_model(self, model_name: str) -> bool:
        """Removes all mappings for a specific model from the lockfile."""
        with self._thread_lock:
            with self._acquire_file_lock():
                data = self._read_raw()
                models = data.get("models", {})
                if model_name in models:
                    del models[model_name]
                    self._write_raw(data)
                    return True
                return False

    def clear(self) -> None:
        """Clears all cached mappings in the lockfile."""
        with self._thread_lock:
            with self._acquire_file_lock():
                self._write_raw({"version": "1.0", "models": {}})


# Global default manager instance
_default_manager = LockFileManager()


def set_default_lockfile_path(path: str | Path) -> None:
    """Configures the default lockfile path used by module-level functions."""
    global _default_manager
    _default_manager = LockFileManager(path)


def get_mapping(
    model_name: str,
    lockfile_path: str | Path | None = None,
) -> dict[str, str]:
    """Module-level reader for cached model field mappings."""
    if lockfile_path is not None:
        return LockFileManager(lockfile_path).get_mapping(model_name)
    return _default_manager.get_mapping(model_name)


def save_mapping(
    model_name: str,
    field_name: str,
    source_path: str,
    healed_by: str,
    lockfile_path: str | Path | None = None,
) -> None:
    """Module-level atomic persistence for a healed field mapping."""
    if lockfile_path is not None:
        LockFileManager(lockfile_path).save_mapping(model_name, field_name, source_path, healed_by)
    else:
        _default_manager.save_mapping(model_name, field_name, source_path, healed_by)


def invalidate_field(
    model_name: str,
    field_name: str,
    lockfile_path: str | Path | None = None,
) -> bool:
    """Module-level invalidation for a drifted field mapping."""
    if lockfile_path is not None:
        return LockFileManager(lockfile_path).invalidate_field(model_name, field_name)
    return _default_manager.invalidate_field(model_name, field_name)
