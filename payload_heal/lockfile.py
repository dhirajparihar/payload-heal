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
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any, Iterator

# Try importing filelock for cross-platform inter-process locking
try:
    from filelock import FileLock  # type: ignore[import-untyped]
except ImportError:
    FileLock = None  # Fallback to OS-specific primitives or threading lock

logger = logging.getLogger("payload_heal.lockfile")

DEFAULT_LOCKFILE_PATH = Path(".payload_heal_lock.json")


def _resolve_model_key(models: dict[str, Any], model_name: str) -> str | None:
    """Resolves a model key in the lockfile models dictionary.

    Supports:
    - Exact match (`myapp.models.Product` or `Product`)
    - Qualified key to legacy short name (`myapp.models.Product` -> `Product`)
    - Short name to qualified key (`Product` -> `myapp.models.Product` if unique)
    """
    if model_name in models:
        return model_name

    # If model_name is qualified, check if legacy short key exists in models
    if "." in model_name:
        short = model_name.split(".")[-1]
        if short in models:
            return short
        return None

    # model_name is a short name (unqualified). Look for matching qualified keys
    matching = [k for k in models if k.split(".")[-1] == model_name]
    if len(matching) == 1:
        return matching[0]

    return None


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
        """Reads and parses the lockfile JSON. Preserves backup on corruption."""
        if not self.lockfile_path.exists():
            return {"version": "1.0", "models": {}}

        try:
            content = self.lockfile_path.read_text(encoding="utf-8")
            if not content.strip():
                return {"version": "1.0", "models": {}}
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("Lockfile root must be a JSON object")
            data.setdefault("version", "1.0")
            if not isinstance(data.get("models"), dict):
                data["models"] = {}
            return data
        except Exception as exc:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            backup_path = self.lockfile_path.with_name(f"{self.lockfile_path.name}.corrupted.{timestamp}.bak")
            try:
                shutil.copy2(self.lockfile_path, backup_path)
                self._rotate_backups(max_backups=5)
                logger.warning(
                    "Lockfile at '%s' is corrupted or invalid (%s). Preserved backup at '%s'. Returning empty lockfile structure.",
                    self.lockfile_path,
                    exc,
                    backup_path,
                )
            except Exception as copy_err:
                logger.warning(
                    "Lockfile at '%s' is corrupted or invalid (%s). Failed to create backup: %s",
                    self.lockfile_path,
                    exc,
                    copy_err,
                )
            return {"version": "1.0", "models": {}}

    def _rotate_backups(self, max_backups: int = 5) -> None:
        """Retains at most `max_backups` recent corrupted backup files, purging older ones."""
        try:
            pattern = f"{self.lockfile_path.name}.corrupted.*.bak"
            parent = self.lockfile_path.parent
            backups = sorted(parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            for old_bak in backups[max_backups:]:
                try:
                    old_bak.unlink()
                except OSError:
                    pass
        except Exception as err:
            logger.debug("Failed to rotate lockfile backups: %s", err)

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
            model_name: The name or qualified identifier of the Pydantic model class.

        Returns:
            Dictionary mapping `field_name` to healed `source_path`.
            Returns `{}` if the model is not found in the lockfile.
        """
        with self._thread_lock:
            with self._acquire_file_lock():
                data = self._read_raw()
                models = data.get("models", {})
                resolved_key = _resolve_model_key(models, model_name)
                if not resolved_key or not isinstance(models.get(resolved_key), dict):
                    return {}

                mapping: dict[str, str] = {}
                for field_name, entry in models[resolved_key].items():
                    if isinstance(entry, dict) and "source_path" in entry:
                        mapping[field_name] = str(entry["source_path"])
                return mapping

    def get_field_info(self, model_name: str, field_name: str) -> dict[str, Any] | None:
        """Returns the full metadata dictionary for a specific healed field."""
        with self._thread_lock:
            with self._acquire_file_lock():
                data = self._read_raw()
                models = data.get("models", {})
                resolved_key = _resolve_model_key(models, model_name)
                if resolved_key and isinstance(models.get(resolved_key), dict):
                    entry = models[resolved_key].get(field_name)
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
            model_name: Name or qualified identifier of the target Pydantic model.
            field_name: Target attribute field on the model.
            source_path: Traversal path in the upstream payload (e.g. `items[0].price`).
            healed_by: Strategy used to heal (`heuristic` or `ai`).
        """
        with self._thread_lock:
            with self._acquire_file_lock():
                data = self._read_raw()
                models = data.setdefault("models", {})
                # If an existing key resolves to this model, update that key
                resolved_key = _resolve_model_key(models, model_name) or model_name
                model_dict = models.setdefault(resolved_key, {})

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
            model_name: Target Pydantic model name or identifier.
            field_name: The field to invalidate.

        Returns:
            True if the field existed and was removed, False otherwise.
        """
        with self._thread_lock:
            with self._acquire_file_lock():
                data = self._read_raw()
                models = data.get("models", {})
                resolved_key = _resolve_model_key(models, model_name)
                if not resolved_key or resolved_key not in models:
                    return False

                model_dict = models[resolved_key]
                if not isinstance(model_dict, dict) or field_name not in model_dict:
                    return False

                del model_dict[field_name]

                # Prune empty model entries
                if not model_dict:
                    del models[resolved_key]

                self._write_raw(data)
                return True

    def remove_model(self, model_name: str) -> bool:
        """Removes all mappings for a specific model from the lockfile."""
        with self._thread_lock:
            with self._acquire_file_lock():
                data = self._read_raw()
                models = data.get("models", {})
                resolved_key = _resolve_model_key(models, model_name)
                if resolved_key and resolved_key in models:
                    del models[resolved_key]
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


def _mp_worker_sync_test(lockfile_path: str, proc_id: int, count: int = 5) -> None:
    """Internal test helper for verifying inter-process synchronization across spawned OS processes."""
    mgr = LockFileManager(lockfile_path)
    model_name = f"ProcModel_{proc_id}"
    for i in range(count):
        field_name = f"field_{i}"
        source_path = f"data.proc_{proc_id}.f_{i}"
        mgr.save_mapping(model_name, field_name, source_path, "heuristic")
        m = mgr.get_mapping(model_name)
        assert field_name in m
    # Invalidate one field to test concurrent invalidations across processes
    mgr.invalidate_field(model_name, "field_0")

