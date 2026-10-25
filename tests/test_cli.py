"""Tests for developer CLI (payload_heal/cli.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from payload_heal.cli import main
from payload_heal.lockfile import LockFileManager


def test_cli_status_and_reset(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    lockfile = tmp_path / ".payload_heal_lock.json"
    mgr = LockFileManager(lockfile)
    mgr.save_mapping("UserModel", "email", "data.user.email", "heuristic")
    mgr.save_mapping("UserModel", "full_name", "data.user.name", "ai")
    mgr.save_mapping("OrderModel", "total", "data.pricing.total", "heuristic")

    # 1. Status command
    code = main(["--lockfile", str(lockfile), "status"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "UserModel" in captured
    assert "data.user.email" in captured
    assert "heuristic" in captured
    assert "ai" in captured

    # 2. Reset specific model
    code = main(["--lockfile", str(lockfile), "reset", "UserModel"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "Successfully reset mappings for model 'UserModel'" in captured
    assert mgr.get_mapping("UserModel") == {}
    assert mgr.get_mapping("OrderModel") == {"total": "data.pricing.total"}

    # 3. Reset all
    code = main(["--lockfile", str(lockfile), "reset"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "Successfully reset all mappings" in captured
    assert mgr.get_mapping("OrderModel") == {}


def test_cli_inspect(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    json_file = tmp_path / "sample.json"
    data = {
        "items": [
            {"id": i, "name": f"Product {i}", "price": 19.99}
            for i in range(50)
        ]
    }
    json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    code = main(["inspect", str(json_file)])
    assert code == 0
    captured = capsys.readouterr().out
    assert "PAYLOAD-HEAL SKELETON INSPECTION" in captured
    assert "reduction" in captured
    assert "items[0].name" in captured
