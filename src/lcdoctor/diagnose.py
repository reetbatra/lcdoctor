"""Turn scan + probe results into findings a person can act on."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .envs import run_probe
from .index import PUBLIC, ExportIndex, rank_candidates
from .scan import ScanResult, Usage

BROKEN = "broken"  # the import fails: the module or the name is gone
NOT_INSTALLED = "not_installed"  # the package is not in this environment at all
IMPORT_ERROR = "import_error"  # the module exists but blows up when imported
DEPRECATED = "deprecated"  # works today, scheduled to go
OK = "ok"

STATUS_ORDER = [BROKEN, NOT_INSTALLED, IMPORT_ERROR, DEPRECATED, OK]

#: How many candidate locations are probed per missing name.
MAX_CANDIDATES = 6


@dataclass
class Location:
    file: str
    line: int
    statement: str


@dataclass
class Relocation:
    """Where a missing name lives now, according to the latest releases."""

    module: str
    name: str | None
    distribution: str | None
    version: str | None
    installed_in_target: bool
    deprecation: dict[str, Any] | None
    path_warnings: list[str] = field(default_factory=list)

    @property
    def target(self) -> str:
        return f"{self.module}.{self.name}" if self.name else self.module

    @property
    def import_line(self) -> str:
        if self.name:
            return f"from {self.module} import {self.name}"
        return f"import {self.module}"


@dataclass
class Finding:
    module: str
    name: str | None
    status: str
    locations: list[Location]
    detail: str | None = None
    deprecation: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    relocation: Relocation | None = None
    other_homes: list[Relocation] = field(default_factory=list)
    searched: bool = False

    @property
    def target(self) -> str:
        return f"{self.module}.{self.name}" if self.name else self.module

    @property
    def import_line(self) -> str:
        if self.name:
            return f"from {self.module} import {self.name}"
        return f"import {self.module}"


@dataclass
class Locator:
    """The "where did it go?" side: an index plus an interpreter to probe candidates in."""

    index: ExportIndex
    python: Path
    probe_result: dict[str, Any] | None = None


@dataclass
class Diagnosis:
    root: str
    files_scanned: int
    environment: dict[str, Any]
    findings: list[Finding]
    syntax_errors: list[dict[str, Any]]
    mode: str
    locator_used: bool

    def count(self, status: str) -> int:
        return sum(1 for f in self.findings if f.status == status)

    @property
    def failing(self) -> int:
        return self.count(BROKEN) + self.count(NOT_INSTALLED) + self.count(IMPORT_ERROR)


def classify(result: dict[str, Any]) -> str:
    status = result["status"]
    if status in ("missing_name", "missing_module"):
        return BROKEN
    if status == NOT_INSTALLED:
        return NOT_INSTALLED
    if status == IMPORT_ERROR:
        return IMPORT_ERROR
    if result.get("deprecation") or result.get("warnings"):
        return DEPRECATED
    return OK


def _group_usages(usages: list[Usage]) -> dict[tuple[str, str | None], list[Location]]:
    grouped: dict[tuple[str, str | None], list[Location]] = {}
    for u in usages:
        grouped.setdefault((u.module, u.name), []).append(Location(u.file, u.line, u.statement))
    return grouped


def _candidate_targets(
    module: str, name: str | None, indexes: list[ExportIndex]
) -> list[tuple[str, str | None]]:
    seen: list[tuple[str, str | None]] = []
    if name is None:
        for idx in indexes:
            for m in idx.find_module(module):
                if (m, None) not in seen:
                    seen.append((m, None))
        return seen[:MAX_CANDIDATES]
    pool = []
    for idx in indexes:
        pool.extend(idx.find(name))
    for c in rank_candidates(module, pool):
        if (c.module, c.name) not in seen:
            seen.append((c.module, c.name))
    out = seen[:MAX_CANDIDATES]
    # `from langchain import hub` names a submodule: look for `<other pkg>.hub` too.
    for idx in indexes:
        for m in idx.find_module(f"{module}.{name}"):
            parent, _, leaf = m.rpartition(".")
            if parent and (parent, leaf) not in out:
                out.append((parent, leaf))
    return out


def _dist_info(top: str, probe: dict[str, Any]) -> tuple[str | None, str | None]:
    dists = probe.get("module_distributions", {}).get(top) or []
    if not dists:
        return None, None
    dist = dists[0]
    return dist, probe.get("distributions", {}).get(dist)


def diagnose(
    scan: ScanResult,
    target_python: Path,
    *,
    mode: str,
    target_index: ExportIndex | None = None,
    locator: Locator | None = None,
    env_label: str | None = None,
) -> Diagnosis:
    grouped = _group_usages(scan.usages)
    targets = [{"module": m, "name": n} for (m, n) in grouped]
    probe = (
        run_probe(target_python, targets)
        if targets
        else {
            "python": None,
            "executable": str(target_python),
            "distributions": {},
            "results": [],
        }
    )
    by_key = {(r["module"], r["name"]): r for r in probe["results"]}

    findings: list[Finding] = []
    for (module, name), locs in grouped.items():
        r = by_key[(module, name)]
        detail = r.get("error")
        if r["status"] == NOT_INSTALLED:
            detail = f"{module.split('.', 1)[0]} is not installed in this environment"
        findings.append(
            Finding(
                module=module,
                name=name,
                status=classify(r),
                locations=locs,
                detail=detail,
                deprecation=r.get("deprecation"),
                warnings=r.get("warnings") or [],
            )
        )

    if locator is not None:
        _relocate(findings, locator, target_index)

    findings.sort(key=lambda f: (STATUS_ORDER.index(f.status), f.target))
    return Diagnosis(
        root=scan.root,
        files_scanned=scan.files_scanned,
        environment={
            "label": env_label,
            "python": probe.get("python"),
            "executable": probe.get("executable"),
            "distributions": probe.get("distributions", {}),
        },
        findings=findings,
        syntax_errors=[e.__dict__ for e in scan.syntax_errors],
        mode=mode,
        locator_used=locator is not None,
    )


def _owner_candidates(
    name: str, owner_top: str, indexes: list[ExportIndex]
) -> list[tuple[str, str | None]]:
    """Public paths for ``name`` inside the package that defines it, shortest first."""
    pool = [c for idx in indexes for c in idx.find(name) if c.module.split(".", 1)[0] == owner_top]
    ranked = sorted(pool, key=lambda c: (c.visibility != PUBLIC, c.module.count("."), c.module))
    out: list[tuple[str, str | None]] = []
    for c in ranked:
        if (c.module, c.name) not in out:
            out.append((c.module, c.name))
    return out[:2]


def _home_key(home: Relocation, defined_in: str | None) -> tuple[int, int, int, int, int, str]:
    owner_top = defined_in.split(".", 1)[0] if defined_in else None
    canonical = owner_top is not None and home.module.split(".", 1)[0] == owner_top
    private = any(seg.startswith("_") for seg in home.module.split("."))
    return (
        1 if home.deprecation else 0,  # a deprecated home is a stopgap
        1
        if home.path_warnings
        else 0,  # importing from this path warns, even if the object is fine
        0 if canonical else 1,  # the package that defines the object, not one that re-exports it
        1 if private else 0,
        home.module.count("."),
        home.module,
    )


def _relocate(findings: list[Finding], locator: Locator, target_index: ExportIndex | None) -> None:
    broken = [f for f in findings if f.status == BROKEN]
    if not broken:
        return
    indexes = [i for i in (target_index, locator.index) if i is not None]
    plan: dict[str, list[tuple[str, str | None]]] = {}
    for f in broken:
        f.searched = True
        plan[f.target] = _candidate_targets(f.module, f.name, indexes)

    by_key: dict[tuple[str, str | None], dict[str, Any]] = {}
    merged: dict[str, Any] = {"distributions": {}, "module_distributions": {}}

    def probe_missing(keys: set[tuple[str, str | None]]) -> None:
        todo = sorted((k for k in keys if k not in by_key), key=lambda c: (c[0], c[1] or ""))
        if not todo:
            return
        result = run_probe(locator.python, [{"module": m, "name": n} for m, n in todo])
        merged["distributions"].update(result.get("distributions", {}))
        merged["module_distributions"].update(result.get("module_distributions", {}))
        for r in result["results"]:
            by_key[(r["module"], r["name"])] = r

    # Phase 1: the ranked candidates. Phase 2: the defining package's own public
    # path, which ranking by path similarity can miss (langchain.prompts ->
    # langchain_core.prompts shares no segment with langchain_classic.prompts).
    probe_missing({c for cands in plan.values() for c in cands})
    for f in broken:
        if f.name is None:
            continue
        owners = {
            by_key[c]["defined_in"].split(".", 1)[0]
            for c in plan[f.target]
            if c in by_key and by_key[c]["status"] == "ok" and by_key[c].get("defined_in")
        }
        for owner in sorted(owners):
            for c in _owner_candidates(f.name, owner, indexes):
                if c not in plan[f.target]:
                    plan[f.target].append(c)
    probe_missing({c for cands in plan.values() for c in cands})
    locator.probe_result = merged

    for f in broken:
        homes: list[tuple[Relocation, str | None]] = []
        for module, name in plan[f.target]:
            r = by_key.get((module, name))
            if r is None or r["status"] != "ok":
                continue
            dist, version = _dist_info(module.split(".", 1)[0], merged)
            dep = r.get("deprecation")
            if not dep and r.get("warnings"):
                dep = {
                    "since": None,
                    "removal": None,
                    "message": r["warnings"][0],
                    "source": "warning",
                }
            homes.append(
                (
                    Relocation(
                        module=module,
                        name=name,
                        distribution=dist,
                        version=version,
                        installed_in_target=bool(target_index and target_index.has_module(module)),
                        deprecation=dep,
                        path_warnings=list(r.get("warnings") or []),
                    ),
                    r.get("defined_in"),
                )
            )
        if not homes:
            continue
        # The object the best non-deprecated home points at decides which package "owns" it.
        live = [d for h, d in homes if not h.deprecation and d]
        owner = live[0] if live else next((d for _, d in homes if d), None)
        ordered = sorted((h for h, _ in homes), key=lambda h: _home_key(h, owner))
        f.relocation, f.other_homes = ordered[0], ordered[1:]
