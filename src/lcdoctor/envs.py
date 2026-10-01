"""Interpreters: find the project's, build throwaway ones with uv, run the probe."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROBE_PATH = Path(__file__).with_name("_probe.py")
FENCE = "@@LCDOCTOR@@"

#: Always installed into the "where did it go?" environment. These are the packages
#: LangChain's own v1 split moved things into, plus the most common chat model partners.
LOCATOR_BASE = (
    "langchain",
    "langchain-core",
    "langchain-classic",
    "langchain-community",
    "langchain-text-splitters",
    "langgraph",
    "langchain-openai",
    "langchain-anthropic",
)


class LcdoctorError(Exception):
    """A problem the user has to act on. The CLI prints it and exits 2."""


def cache_root() -> Path:
    override = os.environ.get("LCDOCTOR_CACHE")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "lcdoctor"


def venv_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def find_project_python(project: Path) -> Path | None:
    """The interpreter of a virtualenv that lives inside the project, if there is one."""
    base = project if project.is_dir() else project.parent
    for name in (".venv", "venv", "env", ".env"):
        candidate = venv_python(base / name)
        if candidate.exists():
            return candidate
    return None


def resolve_python(project: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        found = path if path.exists() else shutil.which(explicit)
        if not found:
            raise LcdoctorError(f"Python interpreter not found: {explicit}")
        return Path(found)
    found = find_project_python(project)
    if found is None:
        raise LcdoctorError(
            f"No virtualenv found in {project} (looked for .venv, venv, env).\n"
            "Pass the interpreter your app runs on: lcdoctor check PATH --python /path/to/python\n"
            "Or check against fresh releases instead: lcdoctor upgrade PATH"
        )
    return found


def require_uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        raise LcdoctorError(
            "This command builds a temporary environment with uv, which is not on PATH.\n"
            "Install it: https://docs.astral.sh/uv/getting-started/installation/"
        )
    return uv


_SPEC_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def spec_name(spec: str) -> str:
    match = _SPEC_NAME.match(spec)
    if not match:
        raise LcdoctorError(f"Not a valid requirement: {spec!r}")
    return match.group(1).lower().replace("_", "-")


def dist_for_module(top_level: str) -> str:
    """LangChain-family distributions are named after their module: langchain_openai -> langchain-openai."""
    return top_level.lower().replace("_", "-")


def merge_specs(defaults: list[str], overrides: list[str]) -> list[str]:
    """Overrides replace a default with the same distribution name, or are added."""
    by_name: dict[str, str] = {spec_name(s): s for s in defaults}
    for spec in overrides:
        by_name[spec_name(spec)] = spec
    return [by_name[k] for k in sorted(by_name)]


@dataclass
class BuiltEnv:
    python: Path
    specs: list[str]
    reused: bool
    path: Path = field(default_factory=Path)


def build_env(
    specs: list[str],
    python_version: str,
    *,
    purpose: str,
    refresh: bool = False,
    log: Any = None,
) -> BuiltEnv:
    """Create (or reuse) a uv virtualenv with ``specs`` installed.

    Unpinned specs resolve to the latest release, so the cache key includes
    today's date: "latest" is re-resolved at most once a day unless ``refresh``.
    """
    uv = require_uv()
    unpinned = any(not re.search(r"[=<>~!]", s) for s in specs)
    day = _dt.date.today().isoformat() if unpinned else ""
    key_src = json.dumps({"specs": sorted(specs), "py": python_version, "day": day})
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    venv = cache_root() / "envs" / f"{purpose}-{key}"
    marker = venv / ".lcdoctor-complete"
    python = venv_python(venv)
    if marker.exists() and python.exists() and not refresh:
        return BuiltEnv(python, specs, reused=True, path=venv)
    if venv.exists():
        shutil.rmtree(venv)
    venv.parent.mkdir(parents=True, exist_ok=True)
    if log:
        log(f"building {purpose} environment (python {python_version}): {' '.join(specs)}")
    _run([uv, "venv", "--quiet", "--python", python_version, str(venv)], "create a virtualenv")
    _run(
        [uv, "pip", "install", "--quiet", "--python", str(python), *specs],
        f"install {' '.join(specs)}",
    )
    marker.write_text(key_src)
    return BuiltEnv(python, specs, reused=False, path=venv)


def _run(cmd: list[str], what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise LcdoctorError(f"uv failed to {what}:\n{detail}")


def run_probe(
    python: Path, targets: list[dict[str, str | None]], timeout: int = 600
) -> dict[str, Any]:
    payload = json.dumps({"targets": targets, "family_prefixes": ["langchain", "langgraph"]})
    env = dict(os.environ)
    # The user's env may set these to "error"; the probe records warnings itself.
    env.pop("PYTHONWARNINGS", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [str(python), "-I", str(PROBE_PATH)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as exc:
        raise LcdoctorError(f"Could not run {python}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise LcdoctorError(f"Probing {python} took longer than {timeout}s") from exc
    match = re.search(re.escape(FENCE) + r"(.*)" + re.escape(FENCE), proc.stdout, re.DOTALL)
    if proc.returncode != 0 or not match:
        detail = (proc.stderr or proc.stdout).strip()[-2000:]
        raise LcdoctorError(f"The probe failed inside {python} (exit {proc.returncode}):\n{detail}")
    result: dict[str, Any] = json.loads(match.group(1))
    return result


def interpreter_version(python: Path) -> str:
    proc = subprocess.run(
        [str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise LcdoctorError(f"Could not run {python}: {proc.stderr.strip()}")
    return proc.stdout.strip()
