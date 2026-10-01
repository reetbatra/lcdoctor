"""Find every LangChain-family import in a project, statically.

Only the import statements are read. Nothing in the user's project is executed:
the probe step imports LangChain's modules, never the user's own.
"""

from __future__ import annotations

import ast
import os
import re
import tomllib
from dataclasses import dataclass
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
    return ScanResult(str(root), len(files), usages, errors)


_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _family_dist(requirement: str) -> str | None:
    match = _REQ_NAME.match(requirement)
    if not match:
        return None
    name = match.group(1).lower().replace("_", "-").replace(".", "-")
    return name if is_family_module(name.replace("-", "_")) else None


def declared_family_distributions(root: Path) -> list[str]:
    """LangChain-family packages named in requirements*.txt or pyproject.toml (pins dropped).

    ``upgrade`` installs these too, so a package the project depends on but never
    imports directly (langchain-community behind ``langchain.vectorstores``) is present
    in the fresh environment exactly as it would be after a real upgrade.
    """
    base = root if root.is_dir() else root.parent
    names: set[str] = set()
    for req in sorted(base.glob("requirements*.txt")):
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line and not line.startswith("-"):
                dist = _family_dist(line)
                if dist:
                    names.add(dist)
    pyproject = base / "pyproject.toml"
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project", {})
        deps = list(project.get("dependencies", []))
        for group in project.get("optional-dependencies", {}).values():
            deps.extend(group)
        poetry = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        deps.extend(poetry.keys())
        for dep in deps:
            dist = _family_dist(dep)
            if dist:
                names.add(dist)
    return sorted(names)
