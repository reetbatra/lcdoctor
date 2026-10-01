"""Static index of what every installed LangChain-family module exports.

Used to answer "this import is gone, where does the name live now?". It reads
source with ``ast`` instead of importing, because importing all of
langchain_community would take minutes and run arbitrary integration code.

A module exports a name when the name is in its ``__all__``, is a key of one of
LangChain's lazy-import tables (``_module_lookup``, ``DEPRECATED_LOOKUP``,
``_dynamic_imports``), or is a public top-level ``def``/``class``.
"""

from __future__ import annotations

import ast
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .scan import is_family_module

#: Bump when the index format or extraction rules change, so stale caches are ignored.
INDEX_VERSION = 3

PUBLIC = "public"  # in __all__ or a lazy-import table
DEFINED = "defined"  # a top-level def/class that is not in __all__

SKIP_PARTS = {"tests", "test", "unit_tests", "integration_tests", "__pycache__"}


@dataclass
class ModuleExports:
    module: str
    public: set[str] = field(default_factory=set)
    defined: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Candidate:
    module: str
    name: str
    visibility: str

    @property
    def target(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass
class ExportIndex:
    modules: dict[str, ModuleExports]

    def __post_init__(self) -> None:
        self._by_name: dict[str, list[Candidate]] = {}
        for mod in self.modules.values():
            for name in mod.public:
                self._by_name.setdefault(name, []).append(Candidate(mod.module, name, PUBLIC))
            for name in mod.defined - mod.public:
                self._by_name.setdefault(name, []).append(Candidate(mod.module, name, DEFINED))

    def find(self, name: str) -> list[Candidate]:
        return list(self._by_name.get(name, []))

    def has_module(self, module: str) -> bool:
        return module in self.modules

    def find_module(self, module: str) -> list[str]:
        """Modules with the same path below the top-level package, e.g.
        ``langchain.chains.llm`` -> ``langchain_classic.chains.llm``."""
        parts = module.split(".")
        if len(parts) < 2:
            return []
        suffix = ".".join(parts[1:])
        return sorted(
            m for m in self.modules if m != module and m.split(".", 1)[-1] == suffix and "." in m
        )

    # -- persistence -------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": INDEX_VERSION,
                "modules": {
                    k: {"public": sorted(v.public), "defined": sorted(v.defined)}
                    for k, v in self.modules.items()
                },
            }
        )

    @classmethod
    def from_json(cls, text: str) -> ExportIndex | None:
        data = json.loads(text)
        if data.get("version") != INDEX_VERSION:
            return None
        return cls(
            {
                k: ModuleExports(k, set(v["public"]), set(v["defined"]))
                for k, v in data["modules"].items()
            }
        )


def _string_items(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [
            e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
    return []


def _dict_keys(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Dict):
        return [
            k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
        ]
    return []


def _is_lookup_table(name: str) -> bool:
    lowered = name.lower()
    return "lookup" in lowered or lowered == "_dynamic_imports"


def extract_exports(source: str, module: str) -> ModuleExports:
    """Pull public names out of one module's source. Raises SyntaxError."""
    tree = ast.parse(source)
    out = ModuleExports(module)
    dicts: dict[str, list[str]] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                out.public.update(_string_items(node.value))
            continue
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                out.defined.add(node.name)
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or value is None:
                continue
            keys = _dict_keys(value)
            if keys:
                dicts[target.id] = keys
                if _is_lookup_table(target.id):
                    out.public.update(keys)
            if target.id == "__all__":
                out.public.update(_string_items(value))
                # __all__ = list(_module_lookup.keys())
                if (
                    isinstance(value, ast.Call)
                    and value.args
                    and isinstance(value.args[0], ast.Call)
                    and isinstance(value.args[0].func, ast.Attribute)
                    and isinstance(value.args[0].func.value, ast.Name)
                ):
                    out.public.update(dicts.get(value.args[0].func.value.id, []))
    out.public = {n for n in out.public if n.isidentifier()}
    return out


def module_name_for(path: Path, site_packages: Path) -> str | None:
    rel = path.relative_to(site_packages).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or any(not p.isidentifier() for p in parts):
        return None
    return ".".join(parts)


def family_package_dirs(site_packages: Path) -> list[Path]:
    if not site_packages.is_dir():
        return []
    # No __init__.py check: langgraph is a namespace package (PEP 420) shared by
    # langgraph, langgraph-prebuilt and langgraph-checkpoint.
    return sorted(
        p
        for p in site_packages.iterdir()
        if p.is_dir() and is_family_module(p.name) and p.name.isidentifier()
    )


def build_index(site_packages_dirs: Iterable[str]) -> ExportIndex:
    modules: dict[str, ModuleExports] = {}
    for sp in site_packages_dirs:
        sp_path = Path(sp)
        for pkg in family_package_dirs(sp_path):
            for dirpath, dirnames, filenames in os.walk(pkg):
                dirnames[:] = [d for d in dirnames if d not in SKIP_PARTS]
                for fname in filenames:
                    if not fname.endswith(".py"):
                        continue
                    path = Path(dirpath) / fname
                    mod = module_name_for(path, sp_path)
                    if mod is None or mod in modules:
                        continue
                    try:
                        modules[mod] = extract_exports(path.read_text(encoding="utf-8"), mod)
                    except (SyntaxError, UnicodeDecodeError, ValueError):
                        # A file that does not parse cannot export anything we can find.
                        modules[mod] = ModuleExports(mod)
    return ExportIndex(modules)


def _segments_shared(a: str, b: str) -> int:
    """How many trailing path segments match below the top-level package."""
    pa, pb = a.split(".")[1:], b.split(".")[1:]
    shared = 0
    for x, y in zip(pa, pb, strict=False):  # paths differ in length
        if x != y:
            break
        shared += 1
    return shared


def rank_candidates(original_module: str, candidates: list[Candidate]) -> list[Candidate]:
    """Best first: public API, mirrors the original path, no private segments, shortest."""

    def key(c: Candidate) -> tuple[int, int, int, int, str]:
        private = any(seg.startswith("_") for seg in c.module.split("."))
        return (
            0 if c.visibility == PUBLIC else 1,
            -_segments_shared(original_module, c.module),
            1 if private else 0,
            c.module.count("."),
            c.module,
        )

    return sorted((c for c in candidates if c.module != original_module), key=key)


def candidate_dicts(cands: list[Candidate]) -> list[dict[str, str]]:
    return [asdict(c) for c in cands]
