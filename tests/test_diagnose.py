"""Diagnosis logic with the probe replaced by canned results (no interpreters, no network)."""

from pathlib import Path
from typing import Any

import pytest

from lcdoctor import diagnose as dmod
from lcdoctor.diagnose import (
    BROKEN,
    DEPRECATED,
    IMPORT_ERROR,
    NOT_INSTALLED,
    OK,
    Locator,
    classify,
    diagnose,
)
from lcdoctor.index import ExportIndex, ModuleExports
from lcdoctor.scan import ScanResult, Usage

DEP_CLASSIC = {
    "since": "0.1.0",
    "removal": {"package": "langchain-classic", "version": "2.0.0"},
    "message": "Use `langchain.agents.create_agent` instead.",
    "source": "docstring",
}


def r(module: str, name: str | None, status: str = "ok", **kw: Any) -> dict[str, Any]:
    base = {
        "module": module,
        "name": name,
        "status": status,
        "kind": None,
        "defined_in": None,
        "error": None,
        "missing_module": None,
        "deprecation": None,
        "warnings": [],
    }
    base.update(kw)
    return base


class FakeProbe:
    """Routes run_probe calls by interpreter path to a table of canned results."""

    def __init__(
        self,
        tables: dict[str, dict[tuple[str, str | None], dict[str, Any]]],
        dists: dict[str, list[str]] | None = None,
    ):
        self.tables = tables
        self.dists = dists or {}
        self.calls: list[tuple[str, list[tuple[str, str | None]]]] = []

    def __call__(
        self, python: Path, targets: list[dict[str, Any]], timeout: int = 600
    ) -> dict[str, Any]:
        keys = [(t["module"], t["name"]) for t in targets]
        self.calls.append((str(python), keys))
        table = self.tables[str(python)]
        results = [
            table.get(k) or r(k[0], k[1], "missing_name", error="unknown to fake") for k in keys
        ]
        return {
            "python": "3.12.0",
            "executable": str(python),
            "site_packages": [],
            "distributions": {
                "langchain": "1.4.3",
                "langchain-core": "1.6.6",
                "langchain-classic": "1.0.8",
            },
            "module_distributions": {
                top: self.dists.get(top, [top.replace("_", "-")])
                for top in {k[0].split(".")[0] for k in keys}
            },
            "results": results,
        }


def scan_of(*usages: tuple[str, str | None, str]) -> ScanResult:
    return ScanResult("/proj", 1, [Usage(f, 1, m, n, f"stmt {m}") for m, n, f in usages], [])


def index_of(**modules: tuple[set[str], set[str]]) -> ExportIndex:
    return ExportIndex(
        {
            k.replace("__", "."): ModuleExports(k.replace("__", "."), pub, dfn)
            for k, (pub, dfn) in modules.items()
        }
    )


def test_classify():
    assert classify(r("a", "b", "missing_name")) == BROKEN
    assert classify(r("a", "b", "missing_module")) == BROKEN
    assert classify(r("a", "b", "not_installed")) == NOT_INSTALLED
    assert classify(r("a", "b", "import_error")) == IMPORT_ERROR
    assert classify(r("a", "b", deprecation=DEP_CLASSIC)) == DEPRECATED
    assert classify(r("a", "b", warnings=["Importing b from a is deprecated"])) == DEPRECATED
    assert classify(r("a", "b")) == OK


def test_groups_locations_and_orders_by_severity(monkeypatch: pytest.MonkeyPatch):
    fake = FakeProbe(
        {
            "/target": {
                ("langchain_core.tools", "tool"): r("langchain_core.tools", "tool"),
                ("langgraph.prebuilt", "create_react_agent"): r(
                    "langgraph.prebuilt", "create_react_agent", deprecation=DEP_CLASSIC
                ),
                ("langchain.chains", "LLMChain"): r(
                    "langchain.chains", "LLMChain", "missing_module"
                ),
                ("langchain_anthropic", "ChatAnthropic"): r(
                    "langchain_anthropic", "ChatAnthropic", "not_installed"
                ),
            }
        }
    )
    monkeypatch.setattr(dmod, "run_probe", fake)
    scan = scan_of(
        ("langchain_core.tools", "tool", "a.py"),
        ("langgraph.prebuilt", "create_react_agent", "a.py"),
        ("langchain.chains", "LLMChain", "a.py"),
        ("langchain.chains", "LLMChain", "b.py"),
        ("langchain_anthropic", "ChatAnthropic", "b.py"),
    )
    d = diagnose(scan, Path("/target"), mode="check")
    assert [f.status for f in d.findings] == [BROKEN, NOT_INSTALLED, DEPRECATED, OK]
    assert [loc.file for loc in d.findings[0].locations] == ["a.py", "b.py"]
    assert d.findings[1].detail == "langchain_anthropic is not installed in this environment"
    assert d.failing == 2
    assert len(fake.calls) == 1  # one probe call for every distinct import


def test_relocation_prefers_defining_package_over_reexport(monkeypatch: pytest.MonkeyPatch):
    # langchain.prompts.PromptTemplate is gone. langchain_classic.prompts re-exports it and
    # mirrors the old path, so it ranks first, but langchain_core defines it.
    idx = index_of(
        langchain_classic__prompts=({"PromptTemplate"}, set()),
        langchain_core__prompts=({"PromptTemplate"}, set()),
        langchain_core__prompts__prompt=(set(), {"PromptTemplate"}),
    )
    loc = {
        ("langchain_classic.prompts", "PromptTemplate"): r(
            "langchain_classic.prompts",
            "PromptTemplate",
            defined_in="langchain_core.prompts.prompt",
        ),
        ("langchain_core.prompts", "PromptTemplate"): r(
            "langchain_core.prompts", "PromptTemplate", defined_in="langchain_core.prompts.prompt"
        ),
        ("langchain_core.prompts.prompt", "PromptTemplate"): r(
            "langchain_core.prompts.prompt",
            "PromptTemplate",
            defined_in="langchain_core.prompts.prompt",
        ),
    }
    fake = FakeProbe(
        {
            "/target": {
                ("langchain.prompts", "PromptTemplate"): r(
                    "langchain.prompts", "PromptTemplate", "missing_module"
                )
            },
            "/locator": loc,
        }
    )
    monkeypatch.setattr(dmod, "run_probe", fake)
    d = diagnose(
        scan_of(("langchain.prompts", "PromptTemplate", "a.py")),
        Path("/target"),
        mode="upgrade",
        locator=Locator(idx, Path("/locator")),
    )
    f = d.findings[0]
    assert f.relocation is not None
    assert f.relocation.import_line == "from langchain_core.prompts import PromptTemplate"
    assert f.relocation.distribution == "langchain-core"
    assert [h.module for h in f.other_homes] == [
        "langchain_core.prompts.prompt",
        "langchain_classic.prompts",
    ]


def test_relocation_second_phase_finds_owner_path_outside_top_candidates(
    monkeypatch: pytest.MonkeyPatch,
):
    # Many mirrored re-exports crowd the first phase; the defining package only shows up in phase two.
    mirrors = {
        f"langchain_classic__agents__m{i}": ({"Tool"}, set()) for i in range(dmod.MAX_CANDIDATES)
    }
    idx = index_of(langchain_core__tools=({"Tool"}, set()), **mirrors)
    table = {
        (f"langchain_classic.agents.m{i}", "Tool"): r(
            f"langchain_classic.agents.m{i}", "Tool", defined_in="langchain_core.tools.simple"
        )
        for i in range(dmod.MAX_CANDIDATES)
    }
    table[("langchain_core.tools", "Tool")] = r(
        "langchain_core.tools", "Tool", defined_in="langchain_core.tools.simple"
    )
    fake = FakeProbe(
        {
            "/target": {
                ("langchain.agents", "Tool"): r("langchain.agents", "Tool", "missing_name")
            },
            "/locator": table,
        }
    )
    monkeypatch.setattr(dmod, "run_probe", fake)
    d = diagnose(
        scan_of(("langchain.agents", "Tool", "a.py")),
        Path("/target"),
        mode="upgrade",
        locator=Locator(idx, Path("/locator")),
    )
    assert d.findings[0].relocation.module == "langchain_core.tools"
    locator_calls = [keys for py, keys in fake.calls if py == "/locator"]
    assert len(locator_calls) == 2
    assert ("langchain_core.tools", "Tool") in locator_calls[1]


def test_relocation_ranks_deprecated_and_warning_paths_last(monkeypatch: pytest.MonkeyPatch):
    idx = index_of(
        langchain_classic=({"LLMChain"}, set()),
        langchain_classic__chains=({"LLMChain"}, set()),
    )
    fake = FakeProbe(
        {
            "/target": {
                ("langchain.chains", "LLMChain"): r(
                    "langchain.chains", "LLMChain", "missing_module"
                )
            },
            "/locator": {
                ("langchain_classic", "LLMChain"): r(
                    "langchain_classic",
                    "LLMChain",
                    deprecation=DEP_CLASSIC,
                    defined_in="langchain_classic.chains.llm",
                    warnings=[
                        "Importing LLMChain from langchain root module is no longer supported."
                    ],
                ),
                ("langchain_classic.chains", "LLMChain"): r(
                    "langchain_classic.chains",
                    "LLMChain",
                    deprecation=DEP_CLASSIC,
                    defined_in="langchain_classic.chains.llm",
                ),
            },
        }
    )
    monkeypatch.setattr(dmod, "run_probe", fake)
    d = diagnose(
        scan_of(("langchain.chains", "LLMChain", "a.py")),
        Path("/target"),
        mode="upgrade",
        locator=Locator(idx, Path("/locator")),
    )
    f = d.findings[0]
    assert f.relocation.module == "langchain_classic.chains"
    assert f.relocation.deprecation == DEP_CLASSIC
    assert f.other_homes[0].path_warnings


def test_broken_with_no_home_is_marked_searched(monkeypatch: pytest.MonkeyPatch):
    fake = FakeProbe(
        {
            "/target": {
                ("langchain.agents", "Gone"): r("langchain.agents", "Gone", "missing_name")
            },
            "/locator": {},
        }
    )
    monkeypatch.setattr(dmod, "run_probe", fake)
    d = diagnose(
        scan_of(("langchain.agents", "Gone", "a.py")),
        Path("/target"),
        mode="upgrade",
        locator=Locator(index_of(), Path("/locator")),
    )
    assert d.findings[0].searched and d.findings[0].relocation is None


def test_installed_in_target_comes_from_target_index(monkeypatch: pytest.MonkeyPatch):
    idx = index_of(langchain_core__prompts=({"PromptTemplate"}, set()))
    fake = FakeProbe(
        {
            "/target": {
                ("langchain.prompts", "PromptTemplate"): r(
                    "langchain.prompts", "PromptTemplate", "missing_module"
                )
            },
            "/locator": {
                ("langchain_core.prompts", "PromptTemplate"): r(
                    "langchain_core.prompts", "PromptTemplate"
                )
            },
        }
    )
    monkeypatch.setattr(dmod, "run_probe", fake)
    scan = scan_of(("langchain.prompts", "PromptTemplate", "a.py"))
    without = diagnose(scan, Path("/target"), mode="check", locator=Locator(idx, Path("/locator")))
    assert without.findings[0].relocation.installed_in_target is False
    with_idx = diagnose(
        scan,
        Path("/target"),
        mode="check",
        target_index=idx,
        locator=Locator(idx, Path("/locator")),
    )
    assert with_idx.findings[0].relocation.installed_in_target is True
