from pathlib import Path

from lcdoctor.index import (
    DEFINED,
    PUBLIC,
    Candidate,
    ExportIndex,
    build_index,
    extract_exports,
    module_name_for,
    rank_candidates,
)


def test_extract_all_and_defs():
    src = (
        "from x import Y\n"
        "__all__ = ['Y', 'helper']\n"
        "__all__ += ('Extra',)\n"
        "def helper(): ...\n"
        "def _private(): ...\n"
        "class Public: ...\n"
        "async def run(): ...\n"
    )
    ex = extract_exports(src, "pkg.mod")
    assert ex.public == {"Y", "helper", "Extra"}
    assert ex.defined == {"helper", "Public", "run"}


def test_extract_lazy_lookup_tables():
    # The shapes LangChain itself uses for lazy imports.
    src = (
        "_module_lookup = {'LLMChain': 'langchain_classic.chains.llm', 'QA': 'x'}\n"
        "DEPRECATED_LOOKUP = {'Tool': 'langchain_community.tools'}\n"
        "_dynamic_imports: dict[str, str] = {'tool': 'tools'}\n"
        "__all__ = list(_module_lookup.keys())\n"
        "settings = {'not': 'exports'}\n"
    )
    ex = extract_exports(src, "langchain_classic.chains")
    assert ex.public == {"LLMChain", "QA", "Tool", "tool"}


def test_module_name_for(tmp_path: Path):
    assert module_name_for(tmp_path / "langchain" / "__init__.py", tmp_path) == "langchain"
    assert module_name_for(tmp_path / "langchain" / "a" / "b.py", tmp_path) == "langchain.a.b"
    assert module_name_for(tmp_path / "langchain" / "not-valid.py", tmp_path) is None


def _write(base: Path, rel: str, text: str) -> None:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_build_index_only_family_packages_and_skips_tests(tmp_path: Path):
    _write(tmp_path, "langchain_classic/__init__.py", "")
    _write(tmp_path, "langchain_classic/chains/__init__.py", "__all__ = ['LLMChain']\n")
    _write(tmp_path, "langchain_classic/chains/llm.py", "class LLMChain: ...\n")
    _write(tmp_path, "langchain_classic/tests/test_x.py", "class LLMChain: ...\n")
    _write(tmp_path, "langchain_classic/bad.py", "def (:\n")
    _write(tmp_path, "requests/__init__.py", "class LLMChain: ...\n")
    idx = build_index([str(tmp_path)])
    assert set(idx.modules) == {
        "langchain_classic",
        "langchain_classic.chains",
        "langchain_classic.chains.llm",
        "langchain_classic.bad",
    }
    assert {c.target for c in idx.find("LLMChain")} == {
        "langchain_classic.chains.LLMChain",
        "langchain_classic.chains.llm.LLMChain",
    }
    assert idx.find_module("langchain.chains.llm") == ["langchain_classic.chains.llm"]
    assert idx.has_module("langchain_classic.chains")


def test_index_json_roundtrip(tmp_path: Path):
    _write(tmp_path, "langgraph/prebuilt/__init__.py", "__all__ = ['create_react_agent']\n")
    idx = build_index([str(tmp_path)])
    again = ExportIndex.from_json(idx.to_json())
    assert again is not None
    assert [c.target for c in again.find("create_react_agent")] == [
        "langgraph.prebuilt.create_react_agent"
    ]
    assert ExportIndex.from_json('{"version": -1, "modules": {}}') is None


def test_rank_prefers_public_mirrored_short_paths():
    cands = [
        Candidate("langchain_community.agents.thing", "Tool", DEFINED),
        Candidate("langchain_classic.agents", "Tool", PUBLIC),
        Candidate("langchain_classic._api.agents", "Tool", PUBLIC),
        Candidate("langchain_core.tools", "Tool", PUBLIC),
        Candidate("langchain.agents", "Tool", PUBLIC),  # the original, never a candidate
    ]
    ranked = [c.module for c in rank_candidates("langchain.agents", cands)]
    assert ranked == [
        "langchain_classic.agents",
        "langchain_core.tools",
        "langchain_classic._api.agents",
        "langchain_community.agents.thing",
    ]
