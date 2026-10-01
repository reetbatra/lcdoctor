"""lcdoctor command line.

    lcdoctor check   [PATH]   are the LangChain imports in PATH OK in the env it runs on today?
    lcdoctor upgrade [PATH]   what breaks if PATH upgrades to the latest (or --to) releases?

Exit codes: 0 nothing failing, 1 failing imports (or deprecations with --strict),
2 lcdoctor could not run (bad path, no interpreter, uv missing, install failed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import __version__
from .diagnose import DEPRECATED, Diagnosis, Locator, diagnose
from .envs import (
    LOCATOR_BASE,
    LcdoctorError,
    build_env,
    cache_root,
    dist_for_module,
    interpreter_version,
    merge_specs,
    resolve_python,
    run_probe,
)
from .fix import apply_fixes, plan_fixes
from .index import ExportIndex, build_index
from .report import render_json, render_markdown, render_text, use_color
from .scan import ScanResult, declared_family_distributions, scan_project


def _log(msg: str) -> None:
    print(f"lcdoctor: {msg}", file=sys.stderr, flush=True)


def load_index(python: Path, *, refresh: bool = False) -> ExportIndex:
    """Index the LangChain-family packages installed for ``python``, cached by their versions."""
    info = run_probe(python, [])
    key_src = json.dumps(
        {"site": info["site_packages"], "dists": info["distributions"]}, sort_keys=True
    )
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    path = cache_root() / "index" / f"{key}.json"
    if path.exists() and not refresh:
        cached = ExportIndex.from_json(path.read_text())
        if cached is not None:
            return cached
    index = build_index(info["site_packages"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(index.to_json())
    return index


def used_distributions(scan: ScanResult) -> list[str]:
    return sorted({dist_for_module(u.top_level) for u in scan.usages})


def build_locator(
    scan: ScanResult, python_version: str, extra: list[str], refresh: bool
) -> Locator:
    specs = merge_specs(list(LOCATOR_BASE) + used_distributions(scan), extra)
    env = build_env(specs, python_version, purpose="locator", refresh=refresh, log=_log)
    return Locator(index=load_index(env.python, refresh=refresh), python=env.python)


def run_check(args: argparse.Namespace) -> Diagnosis:
    root = Path(args.path)
    scan = scan_project(root)
    python = resolve_python(root, args.python)
    locator = None
    if scan.usages and not args.offline:
        locator = build_locator(scan, interpreter_version(python), [], args.refresh)
    target_index = load_index(python, refresh=args.refresh) if locator else None
    return diagnose(
        scan,
        python,
        mode="check",
        target_index=target_index,
        locator=locator,
        env_label=str(python),
    )


def run_upgrade(args: argparse.Namespace) -> Diagnosis:
    root = Path(args.path)
    scan = scan_project(root)
    declared = declared_family_distributions(root)
    specs = merge_specs(sorted(set(used_distributions(scan)) | set(declared)), args.to or [])
    if not scan.usages:
        return diagnose(scan, Path(sys.executable), mode="upgrade", env_label="nothing to install")
    env = build_env(specs, args.python_version, purpose="upgrade", refresh=args.refresh, log=_log)
    locator = (
        None
        if args.offline
        else build_locator(scan, args.python_version, args.to or [], args.refresh)
    )
    target_index = None if args.offline else load_index(env.python, refresh=args.refresh)
    label = "fresh env: " + " ".join(specs)
    return diagnose(
        scan,
        env.python,
        mode="upgrade",
        target_index=target_index,
        locator=locator,
        env_label=label,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lcdoctor",
        description="Find the LangChain imports that break when you upgrade, and where each one lives now.",
    )
    parser.add_argument("--version", action="version", version=f"lcdoctor {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "path", nargs="?", default=".", help="project directory or file (default: .)"
        )
        p.add_argument("--format", choices=["text", "markdown", "json"], default="text")
        p.add_argument(
            "--strict", action="store_true", help="exit 1 on deprecations too, not just failures"
        )
        p.add_argument(
            "--offline",
            action="store_true",
            help="do not build the environment used to find where removed names went",
        )
        p.add_argument(
            "--refresh", action="store_true", help="rebuild cached environments and indexes"
        )
        p.add_argument("--show-ok", action="store_true", help="list imports that are fine, too")
        p.add_argument(
            "--fix",
            action="store_true",
            help="rewrite broken imports to their new location (mechanical moves only)",
        )
        p.add_argument(
            "--diff", action="store_true", help="print the --fix changes without writing them"
        )

    check = sub.add_parser(
        "check", help="check imports against the environment the project runs on"
    )
    common(check)
    check.add_argument(
        "--python", help="interpreter to check against (default: the project's .venv)"
    )

    upgrade = sub.add_parser(
        "upgrade", help="check imports against the latest releases, before upgrading"
    )
    common(upgrade)
    upgrade.add_argument(
        "--to",
        action="append",
        metavar="SPEC",
        help="pin a release, e.g. --to langchain==1.4.3 (repeatable; default: latest)",
    )
    upgrade.add_argument(
        "--python-version", default="3.12", help="Python for the fresh env (default: 3.12)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        diagnosis = run_upgrade(args) if args.command == "upgrade" else run_check(args)
    except (LcdoctorError, FileNotFoundError) as exc:
        print(f"lcdoctor: error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        sys.stdout.write(render_json(diagnosis))
    elif args.format == "markdown":
        sys.stdout.write(render_markdown(diagnosis))
    else:
        sys.stdout.write(render_text(diagnosis, color=use_color(), show_ok=args.show_ok))

    if args.fix or args.diff:
        edits = plan_fixes(diagnosis)
        base = (
            Path(diagnosis.root) if Path(diagnosis.root).is_dir() else Path(diagnosis.root).parent
        )
        for e in edits:
            print(e.diff(str(e.path.relative_to(base))), file=sys.stderr, end="")
        if args.fix:
            apply_fixes(edits)
            moved = sum(len(e.moved) for e in edits)
            _log(
                f"rewrote {moved} import(s) in {len(edits)} file(s)"
                if edits
                else "nothing to rewrite"
            )

    failing = diagnosis.failing
    if args.fix and failing:
        # Rewritten imports are fixed in the source; re-run to confirm instead of guessing.
        _log("re-run lcdoctor to confirm the rewritten imports resolve")
    if failing or (args.strict and diagnosis.count(DEPRECATED)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
