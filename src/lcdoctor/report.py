"""Render a Diagnosis as terminal text, Markdown (for CI summaries and PRs), or JSON."""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from dataclasses import asdict
from typing import Any

from .diagnose import BROKEN, DEPRECATED, IMPORT_ERROR, NOT_INSTALLED, OK, Diagnosis, Finding

LABEL = {
    BROKEN: "BROKEN",
    NOT_INSTALLED: "MISSING PKG",
    IMPORT_ERROR: "IMPORT ERROR",
    DEPRECATED: "DEPRECATED",
    OK: "OK",
}
COLOR = {BROKEN: "31", NOT_INSTALLED: "31", IMPORT_ERROR: "31", DEPRECATED: "33", OK: "32"}


def use_color(stream: Any = None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _paint(text: str, code: str, on: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if on else text


_RST_LINK = re.compile(r"`([^`<]+?)\s*<(https?://[^>]+)>`_")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`>])")


def brief(text: str, sentences: int = 2, limit: int = 260) -> str:
    """First sentences of a long LangChain deprecation message, RST links flattened."""
    text = _RST_LINK.sub(r"\1 (\2)", text).strip()
    parts = _SENTENCE_END.split(text)
    out = " ".join(parts[:sentences])
    if len(out) > limit:
        out = out[: limit - 1].rsplit(" ", 1)[0] + "…"
    return out


def deprecation_sentence(dep: dict[str, Any] | None, *, short: bool = False) -> str:
    if not dep:
        return ""
    bits = []
    if dep.get("since"):
        bits.append(f"deprecated since {dep['since']}")
    else:
        bits.append("deprecated")
    removal = dep.get("removal")
    if removal:
        bits.append(f"removal in {removal['package']}=={removal['version']}")
    head = ", ".join(bits)
    message = (dep.get("message") or "").strip()
    # The removal clause is already in the head; don't say it twice.
    for marker in (" It will be removed in", " It will not be removed until"):
        if marker in message:
            message = message.split(marker)[0].strip()
    if short and message:
        message = brief(message)
    return f"{head}. {message}" if message else head


def _where(f: Finding) -> str:
    locs = [f"{loc.file}:{loc.line}" for loc in f.locations]
    if len(locs) > 4:
        return ", ".join(locs[:4]) + f" (+{len(locs) - 4} more)"
    return ", ".join(locs)


def _home_label(h: Any) -> str:
    pkg = ""
    if h.distribution:
        version = f" {h.version}" if h.version else ""
        installed = "installed" if h.installed_in_target else "not installed"
        pkg = f"   ({h.distribution}{version}, {installed})"
    return f"{h.import_line}{pkg}"


def _wrap(text: str, indent: str, width: int = 100) -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent + "  ")


def env_line(d: Diagnosis) -> str:
    env = d.environment
    dists = env.get("distributions") or {}
    core = [
        f"{k} {v}" for k, v in dists.items() if k in ("langchain", "langchain-core", "langgraph")
    ]
    rest = [
        f"{k} {v}"
        for k, v in dists.items()
        if k not in ("langchain", "langchain-core", "langgraph", "langsmith")
    ]
    shown = core + rest[:4]
    more = f" (+{len(rest) - 4} more)" if len(rest) > 4 else ""
    label = f" [{env['label']}]" if env.get("label") else ""
    return f"python {env.get('python')}{label}: " + (
        ", ".join(shown) + more if shown else "no LangChain packages"
    )


def next_steps(d: Diagnosis) -> list[str]:
    steps: list[str] = []
    installs: dict[str, int] = {}
    fixable = 0
    migrations = 0
    unknown = 0
    for f in d.findings:
        if f.status == BROKEN:
            if f.relocation:
                if f.name and f.relocation.name == f.name:
                    fixable += 1
                if f.relocation.distribution and not f.relocation.installed_in_target:
                    installs[f.relocation.distribution] = (
                        installs.get(f.relocation.distribution, 0) + 1
                    )
                if f.relocation.deprecation:
                    migrations += 1
            else:
                unknown += 1
        elif f.status == DEPRECATED:
            migrations += 1
        elif f.status == NOT_INSTALLED:
            dist = f.module.split(".", 1)[0].replace("_", "-")
            installs[dist] = installs.get(dist, 0) + 1
    for dist, n in sorted(installs.items()):
        steps.append(f"pip install {dist}    # needed by {n} import{'s' if n != 1 else ''}")
    if fixable:
        cmd = "lcdoctor upgrade" if d.mode == "upgrade" else "lcdoctor check"
        steps.append(
            f"{cmd} {_display_root(d)} --fix    # rewrites {fixable} import{'s' if fixable != 1 else ''} to "
            "their new home (mechanical moves only)"
        )
    if migrations:
        steps.append(
            f"{migrations} import{'s' if migrations != 1 else ''} still need a real migration; "
            "see the 'deprecated' lines"
        )
    if unknown:
        steps.append(
            f"{unknown} broken import{'s' if unknown != 1 else ''} with no new home in the latest "
            "releases: removed outright, or in a package lcdoctor did not index"
        )
    return steps


def _display_root(d: Diagnosis) -> str:
    cwd = os.getcwd()
    if d.root == cwd:
        return "."
    if d.root.startswith(cwd + os.sep):
        return d.root[len(cwd) + 1 :]
    return d.root


def render_text(d: Diagnosis, *, color: bool = False, show_ok: bool = False) -> str:
    out: list[str] = []
    title = "what breaks if you upgrade" if d.mode == "upgrade" else "LangChain import check"
    out.append(_paint(f"lcdoctor · {title}", "1", color))
    total = len(d.findings)
    out.append(
        f"scanned {d.files_scanned} Python files in {_display_root(d)}, {total} distinct LangChain imports"
    )
    out.append(env_line(d))
    out.append("")
    counts = [
        _paint(f"{d.count(s)} {LABEL[s].lower()}", COLOR[s], color and d.count(s) > 0)
        for s in (BROKEN, NOT_INSTALLED, IMPORT_ERROR, DEPRECATED, OK)
        if d.count(s) or s in (BROKEN, OK)
    ]
    out.append("  " + "   ".join(counts))

    for f in d.findings:
        if f.status == OK and not show_ok:
            continue
        out.append("")
        tag = _paint(f"{LABEL[f.status]:<12}", COLOR[f.status], color)
        out.append(f"{tag} {f.import_line}")
        out.append(f"             {_where(f)}")
        if f.status in (BROKEN, IMPORT_ERROR, NOT_INSTALLED) and f.detail:
            out.append(_wrap(f"why:  {f.detail}", "             "))
        if f.status == DEPRECATED:
            text = deprecation_sentence(f.deprecation, short=True) if f.deprecation else ""
            if not text and f.warnings:
                text = brief(f.warnings[0])
            out.append(_wrap(f"note: {text}", "             "))
        if f.relocation:
            out.append(_paint(f"             now:  {_home_label(f.relocation)}", "36", color))
            if f.relocation.deprecation:
                out.append(
                    _wrap(
                        f"but:  {deprecation_sentence(f.relocation.deprecation, short=True)}",
                        "             ",
                    )
                )
            for h in f.other_homes[:2]:
                out.append(f"             also: {_home_label(h)}")
        elif f.status == BROKEN and f.searched:
            out.append("             now:  not found in the latest LangChain-family releases")
        elif f.status == BROKEN and not d.locator_used:
            out.append("             now:  (search skipped: --offline)")

    if d.syntax_errors:
        out.append("")
        out.append(f"skipped {len(d.syntax_errors)} file(s) that do not parse:")
        for e in d.syntax_errors[:5]:
            out.append(f"  {e['file']}:{e['line']}  {e['message']}")

    steps = next_steps(d)
    if steps:
        out.append("")
        out.append(_paint("next", "1", color))
        out.extend(f"  {s}" for s in steps)
    elif not d.findings:
        out.append("")
        out.append("no LangChain or LangGraph imports found")
    else:
        out.append("")
        out.append(
            _paint(
                "nothing to do: every LangChain import resolves and none are deprecated",
                "32",
                color,
            )
        )
    return "\n".join(out) + "\n"


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(d: Diagnosis) -> str:
    title = "What breaks if you upgrade" if d.mode == "upgrade" else "LangChain import check"
    lines = [f"### lcdoctor: {title}", ""]
    lines.append(f"`{env_line(d)}`")
    lines.append("")
    lines.append(
        f"**{d.count(BROKEN) + d.count(NOT_INSTALLED) + d.count(IMPORT_ERROR)} failing**, "
        f"**{d.count(DEPRECATED)} deprecated**, {d.count(OK)} ok, across {d.files_scanned} files"
    )
    rows = [f for f in d.findings if f.status != OK]
    if rows:
        lines += ["", "| Status | Import | Where | Now / Note |", "|---|---|---|---|"]
        for f in rows:
            if f.relocation:
                note = f"`{f.relocation.import_line}`"
                if f.relocation.distribution and not f.relocation.installed_in_target:
                    note += f" (needs `{f.relocation.distribution}`)"
                if f.relocation.deprecation:
                    note += f"; {deprecation_sentence(f.relocation.deprecation)}"
            elif f.status == DEPRECATED:
                note = deprecation_sentence(f.deprecation) or "; ".join(f.warnings)
            else:
                note = f.detail or ""
            lines.append(
                f"| {LABEL[f.status]} | `{_md_escape(f.import_line)}` | {_md_escape(_where(f))} | {_md_escape(note)} |"
            )
    steps = next_steps(d)
    if steps:
        lines += ["", "**Next**", ""] + [
            f"- `{s}`" if s.startswith(("pip", "lcdoctor")) else f"- {s}" for s in steps
        ]
    return "\n".join(lines) + "\n"


def render_json(d: Diagnosis) -> str:
    data = asdict(d)
    data["summary"] = {s: d.count(s) for s in (BROKEN, NOT_INSTALLED, IMPORT_ERROR, DEPRECATED, OK)}
    for f, raw in zip(d.findings, data["findings"], strict=True):
        raw["target"] = f.target
        raw["import_line"] = f.import_line
    return json.dumps(data, indent=2) + "\n"
