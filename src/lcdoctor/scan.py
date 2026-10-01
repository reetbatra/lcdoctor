"""Find every LangChain-family import in a project, statically.

Only the import statements are read. Nothing in the user's project is executed:
the probe step imports LangChain's modules, never the user's own.
"""

from __future__ import annotations

import ast
import os
import re
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

#: Top-level module prefixes that belong to the LangChain ecosystem.
FAMILY_PREFIXES = ("langchain", "langgraph")

#: Directories that never contain the user's own source.
SKIP_DIRS = {
    ".git",
    ".hg",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "build",
    "dist",
    "site-packages",
    ".ipynb_checkpoints",
}


@dataclass(frozen=True)
class Usage:
    """One imported name. ``name`` is None for a bare ``import langchain.x``."""

    file: str
    line: int
    module: str
    name: str | None
    statement: str

    @property
    def target(self) -> str:
        return f"{self.module}.{self.name}" if self.name else self.module

    @property
    def top_level(self) -> str:
        return self.module.split(".", 1)[0]


@dataclass(frozen=True)
class SyntaxProblem:
    file: str
    line: int
    message: str


@dataclass
class ScanResult:
    root: str
    files_scanned: int
    usages: list[Usage]
    syntax_errors: list[SyntaxProblem]
    #: Family-named packages that live inside the project (its own code, not a dependency).
    local_packages: list[str] = field(default_factory=list)


def is_family_module(module: str) -> bool:
    top = module.split(".", 1)[0]
    return any(top == p or top.startswith(p + "_") for p in FAMILY_PREFIXES)


def iter_python_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".py" else []
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")
        )
        out.extend(Path(dirpath) / f for f in sorted(filenames) if f.endswith(".py"))
    return out


def _statement_text(source_lines: list[str], node: ast.stmt) -> str:
    end = node.end_lineno or node.lineno
    text = " ".join(line.strip() for line in source_lines[node.lineno - 1 : end])
    return " ".join(text.split())


def scan_source(source: str, rel_path: str) -> list[Usage]:
    """Return the LangChain-family imports in one file. Raises SyntaxError."""
    with warnings.catch_warnings():
        # Invalid escapes etc. in the scanned project's code are its business, not ours.
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source, filename=rel_path)
    lines = source.splitlines()
    usages: list[Usage] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # Relative imports are the user's own modules.
            if node.level or not node.module or not is_family_module(node.module):
                continue
            stmt = _statement_text(lines, node)
            for alias in node.names:
                if alias.name == "*":
                    usages.append(Usage(rel_path, node.lineno, node.module, None, stmt))
                else:
                    usages.append(Usage(rel_path, node.lineno, node.module, alias.name, stmt))
        elif isinstance(node, ast.Import):
            stmt = _statement_text(lines, node)
            for alias in node.names:
                if is_family_module(alias.name):
                    usages.append(Usage(rel_path, node.lineno, alias.name, None, stmt))
    usages.sort(key=lambda u: (u.line, u.target))
    return usages


def local_top_level_names(files: list[Path]) -> set[str]:
    """Top-level module names the project itself provides: package dirs and loose modules.

    ``langchain_chatchat`` is the Langchain-Chatchat repo's own package; treating it as
    a LangChain dependency would install that project's (old) PyPI release.
    """
    names: set[str] = set()
    for path in files:
        # Only importable as a *top-level* name when the containing directory is a
        # source root, not itself a package: pkg/extra/langchain/ is pkg.extra.langchain.
        if path.name == "__init__.py":
            if not (path.parent.parent / "__init__.py").exists():
                names.add(path.parent.name)
        elif not (path.parent / "__init__.py").exists():
            names.add(path.stem)
    return {n for n in names if is_family_module(n)}


def scan_project(root: Path) -> ScanResult:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    base = root if root.is_dir() else root.parent
    files = iter_python_files(root)
    usages: list[Usage] = []
    errors: list[SyntaxProblem] = []
    for path in files:
        rel = str(path.relative_to(base))
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            errors.append(SyntaxProblem(rel, 0, f"not valid UTF-8: {exc}"))
            continue
        try:
            usages.extend(scan_source(source, rel))
        except SyntaxError as exc:
            errors.append(SyntaxProblem(rel, exc.lineno or 0, exc.msg))
    local = local_top_level_names(files)
    usages = [u for u in usages if u.top_level not in local]
    return ScanResult(str(root), len(files), usages, errors, sorted(local))


_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _family_dist(requirement: str) -> str | None:
    match = _REQ_NAME.match(requirement)
    if not match:
        return None
    name = match.group(1).lower().replace("_", "-").replace(".", "-")
    return name if is_family_module(name.replace("-", "_")) else None


def _dependency_files(base: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")
        )
        for f in sorted(filenames):
            if f == "pyproject.toml" or (f.startswith("requirements") and f.endswith(".txt")):
                found.append(Path(dirpath) / f)
    return found


def declared_family_distributions(root: Path, exclude: set[str] | None = None) -> list[str]:
    """LangChain-family packages named in requirements*.txt or pyproject.toml anywhere in
    the project (pins dropped), minus the project's own package names.

    ``upgrade`` installs these too, so a package the project depends on but never
    imports directly (langchain-community behind ``langchain.vectorstores``) is present
    in the fresh environment exactly as it would be after a real upgrade.
    """
    base = root if root.is_dir() else root.parent
    names: set[str] = set()
    own: set[str] = {n.replace("_", "-") for n in (exclude or set())}
    for path in _dependency_files(base):
        if path.suffix == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if line and not line.startswith("-"):
                    dist = _family_dist(line)
                    if dist:
                        names.add(dist)
            continue
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        project = data.get("project", {})
        poetry_tool = data.get("tool", {}).get("poetry", {})
        for own_name in (project.get("name"), poetry_tool.get("name")):
            if isinstance(own_name, str):
                own.add(own_name.lower().replace("_", "-").replace(".", "-"))
        deps = list(project.get("dependencies", []))
        for group in project.get("optional-dependencies", {}).values():
            deps.extend(group)
        for dep_name, spec in poetry_tool.get("dependencies", {}).items():
            # A path/git dependency is local code, not a release to upgrade to.
            if not (isinstance(spec, dict) and ("path" in spec or "git" in spec)):
                deps.append(dep_name)
        for dep in deps:
            dist = _family_dist(dep)
            if dist:
                names.add(dist)
    return sorted(names - own)
