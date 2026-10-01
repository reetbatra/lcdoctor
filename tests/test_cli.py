import sys
from pathlib import Path

import pytest

from lcdoctor import cli, envs


def test_check_without_a_virtualenv_exits_2_with_instructions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    (tmp_path / "a.py").write_text("from langchain.chains import LLMChain\n")
    assert cli.main(["check", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "No virtualenv found" in err and "--python" in err


def test_check_unknown_interpreter_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert cli.main(["check", str(tmp_path), "--python", "/no/such/python"]) == 2
    assert "interpreter not found" in capsys.readouterr().err


def test_missing_path_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert cli.main(["upgrade", str(tmp_path / "nope")]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_project_without_langchain_is_clean(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    (tmp_path / "a.py").write_text("import json\n")
    assert cli.main(["check", str(tmp_path), "--python", sys.executable, "--offline"]) == 0
    assert "no LangChain or LangGraph imports found" in capsys.readouterr().out
    assert cli.main(["upgrade", str(tmp_path)]) == 0


def test_check_reports_not_installed_and_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # The test interpreter has no LangChain at all, so this needs no network.
    (tmp_path / "a.py").write_text("from langchain_definitely_not_installed import X\n")
    code = cli.main(
        ["check", str(tmp_path), "--python", sys.executable, "--offline", "--format", "markdown"]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "| MISSING PKG | `from langchain_definitely_not_installed import X` |" in out


def test_upgrade_without_uv_is_an_explicit_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    (tmp_path / "a.py").write_text("from langchain.chains import LLMChain\n")
    monkeypatch.setattr(envs.shutil, "which", lambda _name: None)
    assert cli.main(["upgrade", str(tmp_path)]) == 2
    assert "uv" in capsys.readouterr().err


def test_merge_specs_overrides_by_name():
    assert envs.merge_specs(
        ["langchain", "langgraph"], ["LangChain==1.4.3", "langchain_openai"]
    ) == [
        "LangChain==1.4.3",
        "langchain_openai",
        "langgraph",
    ]
    with pytest.raises(envs.LcdoctorError):
        envs.spec_name("==1.0")
