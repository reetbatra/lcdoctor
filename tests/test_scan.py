from pathlib import Path

import pytest

from lcdoctor.scan import declared_family_distributions, is_family_module, scan_project, scan_source


def test_family_module_matching():
    assert is_family_module("langchain")
    assert is_family_module("langchain.agents")
    assert is_family_module("langchain_openai.chat_models")
    assert is_family_module("langgraph.prebuilt")
    assert not is_family_module("langchainx")  # a different package that shares a prefix
    assert not is_family_module("openai")
    assert not is_family_module("my_langchain_helpers")


def test_from_imports_one_usage_per_name():
    src = "from langchain.agents import AgentType, initialize_agent as init\n"
    usages = scan_source(src, "a.py")
    assert [(u.module, u.name, u.line) for u in usages] == [
        ("langchain.agents", "AgentType", 1),
        ("langchain.agents", "initialize_agent", 1),
    ]
    assert usages[0].statement == "from langchain.agents import AgentType, initialize_agent as init"


def test_plain_and_star_imports_are_module_level():
    src = "import langchain.chains\nimport os, langgraph\nfrom langchain.prompts import *\n"
    got = [(u.module, u.name) for u in scan_source(src, "a.py")]
    assert got == [("langchain.chains", None), ("langgraph", None), ("langchain.prompts", None)]


def test_ignores_relative_and_unrelated_imports():
    src = "from . import langchain\nfrom .langchain import x\nimport requests\nfrom openai import OpenAI\n"
    assert scan_source(src, "a.py") == []


def test_finds_imports_nested_in_functions_and_try_blocks():
    src = (
        "def build():\n"
        "    try:\n"
        "        from langchain.chat_models import ChatOpenAI\n"
        "    except ImportError:\n"
        "        from langchain_openai import ChatOpenAI\n"
    )
    got = [(u.module, u.line) for u in scan_source(src, "a.py")]
    assert got == [("langchain.chat_models", 3), ("langchain_openai", 5)]


def test_multiline_statement_is_collapsed():
    src = "from langchain.agents import (\n    AgentType,\n    initialize_agent,\n)\n"
    usages = scan_source(src, "a.py")
    assert usages[0].statement == "from langchain.agents import ( AgentType, initialize_agent, )"


def test_scan_project_skips_virtualenvs_and_reports_syntax_errors(tmp_path: Path):
    (tmp_path / "app.py").write_text("from langchain.chains import LLMChain\n")
    (tmp_path / "broken.py").write_text("def nope(:\n")
    venv_pkg = tmp_path / ".venv" / "lib" / "site-packages" / "langchain"
    venv_pkg.mkdir(parents=True)
    (venv_pkg / "__init__.py").write_text("from langchain.chains import LLMChain\n")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "tools.py").write_text("from langgraph.prebuilt import create_react_agent\n")

    result = scan_project(tmp_path)
    assert result.files_scanned == 3
    assert sorted(u.file for u in result.usages) == ["app.py", "pkg/tools.py"]
    assert [e.file for e in result.syntax_errors] == ["broken.py"]


def test_scan_project_single_file(tmp_path: Path):
    f = tmp_path / "one.py"
    f.write_text("import langchain\n")
    result = scan_project(f)
    assert result.files_scanned == 1
    assert result.usages[0].file == "one.py"


def test_scan_project_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        scan_project(tmp_path / "nope")


def test_declared_distributions_from_requirements_and_pyproject(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text(
        "# pinned before 1.0\nlangchain==0.3.30\nlangchain_community>=0.3  # comment\n"
        "-r other.txt\nrequests\nlanggraph[extra]~=0.6\n"
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["langchain-openai>=0.3", "httpx"]\n'
        '[project.optional-dependencies]\nanthropic = ["langchain-anthropic"]\n'
    )
    assert declared_family_distributions(tmp_path) == [
        "langchain",
        "langchain-anthropic",
        "langchain-community",
        "langchain-openai",
        "langgraph",
    ]


def test_declared_distributions_poetry(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.poetry.dependencies]\npython = "^3.11"\nlangchain-core = "^0.3"\n'
    )
    assert declared_family_distributions(tmp_path) == ["langchain-core"]


def test_project_own_family_named_package_is_not_a_dependency(tmp_path: Path):
    # Regression: Langchain-Chatchat ships its own `langchain_chatchat` package.
    pkg = tmp_path / "libs" / "server" / "langchain_chatchat"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (tmp_path / "langchain_helpers.py").write_text("")
    (tmp_path / "app.py").write_text(
        "import langchain_chatchat\nfrom langchain_helpers import x\nfrom langchain.chains import LLMChain\n"
    )
    result = scan_project(tmp_path)
    assert [u.module for u in result.usages] == ["langchain.chains"]
    assert result.local_packages == ["langchain_chatchat", "langchain_helpers"]


def test_declared_distributions_walk_tree_and_drop_own_and_path_deps(tmp_path: Path):
    server = tmp_path / "libs" / "server"
    server.mkdir(parents=True)
    (server / "pyproject.toml").write_text(
        '[tool.poetry]\nname = "langchain-chatchat"\n'
        '[tool.poetry.dependencies]\nlangchain = "0.1.17"\nlangchain-community = "0.0.36"\n'
        'langchain-local = { path = "../local" }\nlangchainhub = "0.1.14"\n'
    )
    (tmp_path / "pyproject.toml").write_text(
        '[tool.poetry]\nname = "Chatchat"\n[tool.poetry.dependencies]\nlangchain-chatchat = "*"\n'
    )
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "requirements.txt").write_text("langchain-ignored\n")
    assert declared_family_distributions(tmp_path) == ["langchain", "langchain-community"]


def test_syntax_warnings_in_scanned_code_are_silenced(recwarn: pytest.WarningsRecorder):
    scan_source('import re\nPAT = "\\s+"\nfrom langchain.chains import LLMChain\n', "a.py")
    assert not [w for w in recwarn if issubclass(w.category, SyntaxWarning)]


def test_nested_family_named_subpackage_is_not_local_top_level(tmp_path: Path):
    # Regression: Langchain-Chatchat has open_chatcaht/extra/langchain/, which is not `langchain`.
    nested = tmp_path / "sdk" / "open_chatcaht" / "extra" / "langchain"
    nested.mkdir(parents=True)
    for d in (nested, nested.parent, nested.parent.parent):
        (d / "__init__.py").write_text("")
    (nested.parent / "langchain_tools.py").write_text("")
    (tmp_path / "app.py").write_text("from langchain.chains import LLMChain\n")
    result = scan_project(tmp_path)
    assert result.local_packages == []
    assert [u.module for u in result.usages] == ["langchain.chains"]
