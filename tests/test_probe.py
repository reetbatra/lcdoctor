"""The probe runs inside the user's interpreter; these test it in-process on fake packages."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from lcdoctor import _probe
from lcdoctor.envs import PROBE_PATH

V1_DOC = """Run a chain.

!!! deprecated "0.1.17 Use `RunnableSequence, e.g., `prompt | llm`` instead. It will be removed in langchain-classic==2.0.0."

More text.
"""

V03_DOC = """Chat model.

.. deprecated:: 0.0.10
   Use :class:`~langchain_openai.ChatOpenAI` instead. It will not be removed until langchain-community==1.0.

Args:
    x: y
"""


def test_parse_admonition_docstring():
    info = _probe.parse_deprecation_doc(V1_DOC)
    assert info["since"] == "0.1.17"
    assert info["removal"] == {"package": "langchain-classic", "version": "2.0.0"}
    assert info["message"].startswith("Use `RunnableSequence")


def test_parse_sphinx_docstring():
    info = _probe.parse_deprecation_doc(V03_DOC)
    assert info["since"] == "0.0.10"
    assert info["removal"] == {"package": "langchain-community", "version": "1.0"}
    assert "`langchain_openai.ChatOpenAI`" in info["message"]
    assert ":class:" not in info["message"]


def test_parse_ignores_docstrings_without_deprecation():
    assert _probe.parse_deprecation_doc("Just a normal docstring.") is None
    assert _probe.parse_deprecation_doc(None) is None


@pytest.fixture
def fake_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pkg = tmp_path / "langchain_fakepkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        textwrap.dedent('''
        import warnings

        def current():
            """Fine."""

        def old():
            """Old.

            !!! deprecated "0.2.0 Use `current` instead. It will be removed in langchain-fakepkg==1.0.0."
            """
        old.__deprecated__ = "Use current instead."

        def __getattr__(name):
            if name == "Moved":
                warnings.warn("Importing Moved from langchain root module is no longer supported.")
                return current
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    ''')
    )
    (pkg / "sub" / "__init__.py").write_text("")
    (pkg / "needs_dep.py").write_text("import a_dependency_that_is_not_installed\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "langchain_fakepkg"
    for name in [m for m in sys.modules if m.startswith("langchain_fakepkg")]:
        del sys.modules[name]


def test_probe_statuses(fake_family):
    cache: dict = {}
    ok = _probe.probe_one(fake_family, "current", cache)
    assert ok["status"] == "ok" and ok["deprecation"] is None
    assert ok["kind"] == "function" and ok["defined_in"] == fake_family

    old = _probe.probe_one(fake_family, "old", cache)
    assert old["deprecation"]["since"] == "0.2.0"
    assert old["deprecation"]["source"] == "__deprecated__+docstring"

    moved = _probe.probe_one(fake_family, "Moved", cache)
    assert moved["status"] == "ok"
    assert any("no longer supported" in w for w in moved["warnings"])

    missing = _probe.probe_one(fake_family, "gone", cache)
    assert missing["status"] == "missing_name"

    submodule = _probe.probe_one(fake_family + ".nope", "x", cache)
    assert submodule["status"] == "missing_module"

    package = _probe.probe_one("langchain_not_a_real_package", "x", cache)
    assert package["status"] == "not_installed"

    dependency = _probe.probe_one(fake_family + ".needs_dep", "x", cache)
    assert dependency["status"] == "import_error"
    assert dependency["missing_module"] == "a_dependency_that_is_not_installed"


def test_probe_script_runs_standalone_and_fences_output():
    payload = json.dumps(
        {"targets": [{"module": "json", "name": "dumps"}, {"module": "json", "name": "nope"}]}
    )
    proc = subprocess.run(
        [sys.executable, "-I", str(PROBE_PATH)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    body = proc.stdout.split("@@LCDOCTOR@@")[1]
    data = json.loads(body)
    assert [r["status"] for r in data["results"]] == ["ok", "missing_name"]
    assert data["python"].count(".") == 2
    assert data["site_packages"]


def test_probe_from_import_of_unimported_submodule(fake_family):
    # `from langchain_fakepkg import sub` works in Python even though sub was never imported.
    cache: dict = {}
    res = _probe.probe_one(fake_family, "sub", cache)
    assert res["status"] == "ok" and res["kind"] == "module"
    missing = _probe.probe_one(fake_family, "nothing_here", cache)
    assert missing["status"] == "missing_name"
