from __future__ import annotations

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from selfrl_chess.cli import app


def test_info_command_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "knightmare" in result.stdout


def test_version_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "knightmare" in result.stdout
