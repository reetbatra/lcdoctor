"""Rewrite broken ``from X import Y`` statements to where Y lives now.

Only mechanical moves are applied: the same object, importable from a new
path. Migrating to a different API (``initialize_agent`` -> ``create_agent``)
changes behaviour, so it is reported, never rewritten.
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass
from pathlib import Path

from .diagnose import BROKEN, Diagnosis


@dataclass
class FileEdit:
    path: Path
    before: str
    after: str
    moved: list[tuple[str, str]]  # (old target, new target)

    def diff(self, rel: str) -> str:
        return "".join(
            difflib.unified_diff(
                self.before.splitlines(keepends=True),
                self.after.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )


def moves_from(diagnosis: Diagnosis) -> dict[tuple[str, str], str]:
    """(old module, name) -> new module, for every broken name with a known home."""
    out: dict[tuple[str, str], str] = {}
    for f in diagnosis.findings:
        if f.status == BROKEN and f.name and f.relocation and f.relocation.name == f.name:
            out[(f.module, f.name)] = f.relocation.module
    return out


def _render(module: str, names: list[ast.alias], indent: str) -> str:
    parts = [f"{a.name} as {a.asname}" if a.asname else a.name for a in names]
    one_line = f"{indent}from {module} import {', '.join(parts)}"
    if len(one_line) <= 100:
        return one_line
    body = "".join(f"{indent}    {p},\n" for p in parts)
    return f"{indent}from {module} import (\n{body}{indent})"


def rewrite_source(
    source: str, moves: dict[tuple[str, str], str]
) -> tuple[str, list[tuple[str, str]]]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []  # (start line idx, end line idx exclusive, replacement)
    moved: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
            continue
        if not any((node.module, a.name) in moves for a in node.names):
            continue
        start, end = node.lineno - 1, (node.end_lineno or node.lineno)
        first = lines[start]
        # Only rewrite statements that own their lines; "x = 1; from a import b" is left alone.
        prefix = first[: node.col_offset]
        trailing = lines[end - 1][node.end_col_offset or 0 :].strip()
        if prefix.strip() or (trailing and not trailing.startswith("#")):
            continue
        groups: dict[str, list[ast.alias]] = {}
        for alias in node.names:
            new_module = moves.get((node.module, alias.name), node.module)
            groups.setdefault(new_module, []).append(alias)
            if new_module != node.module:
                moved.append((f"{node.module}.{alias.name}", f"{new_module}.{alias.name}"))
        newline = "\r\n" if first.endswith("\r\n") else "\n"
        rendered = newline.join(_render(m, names, prefix) for m, names in groups.items())
        if trailing:  # keep a trailing "# noqa" style comment
            rendered += f"  {trailing}"
        rendered += newline
        edits.append((start, end, rendered))
    for start, end, text in sorted(edits, reverse=True):
        lines[start:end] = [text]
    return "".join(lines), moved


def plan_fixes(diagnosis: Diagnosis) -> list[FileEdit]:
    moves = moves_from(diagnosis)
    if not moves:
        return []
    root = Path(diagnosis.root)
    base = root if root.is_dir() else root.parent
    files = sorted(
        {loc.file for f in diagnosis.findings if f.status == BROKEN for loc in f.locations}
    )
    edits: list[FileEdit] = []
    for rel in files:
        path = base / rel
        before = path.read_text(encoding="utf-8")
        after, moved = rewrite_source(before, moves)
        if after != before:
            edits.append(FileEdit(path, before, after, moved))
    return edits


def apply_fixes(edits: list[FileEdit]) -> None:
    for e in edits:
        e.path.write_text(e.after, encoding="utf-8")
