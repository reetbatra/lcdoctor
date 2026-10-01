# What this project is
lcdoctor: a Python CLI that finds LangChain/LangGraph imports that break on upgrade and where each name lives now. It reads LangChain's own runtime metadata (`__deprecated__`, docstring admonitions, import warnings, `obj.__module__`) instead of a hand-written migration table. Portfolio project for AI/DevRel roles: quality bar is "public tool", not "side project".

# Stack
- Python 3.11+, zero runtime dependencies (stdlib only), uv for everything, hatchling build
- pytest, ruff (format + lint), mypy strict

# Commands
- Unit tests (no network): `uv run pytest`
- Integration (builds real envs from PyPI, ~1 min): `uv run pytest -m integration`
- Before shipping: `uv run ruff format src tests && uv run ruff check src tests && uv run mypy && uv run pytest`
- Try it: `uv run lcdoctor upgrade examples/support-agent`

# Layout (src/lcdoctor/)
- `scan.py` AST scan of the user's project; skips the project's own `langchain_*` packages; reads declared deps
- `_probe.py` runs INSIDE the target interpreter as a standalone script: stdlib only, Python 3.8 syntax, no lcdoctor imports
- `envs.py` interpreter discovery, uv-built cached envs (`~/.cache/lcdoctor`), runs the probe
- `index.py` static export index of installed LangChain-family packages (`__all__`, lazy-import tables, defs)
- `diagnose.py` findings + two-phase relocation (ranked candidates, then the defining package's public path)
- `fix.py` mechanical `from X import Y` rewrites only; `report.py` text/markdown/json; `cli.py`

# Rules
- Never add a hand-written "old path -> new path" table. Every answer must come from the installed packages.
- Never silently fall back: no interpreter found, uv missing, or a failed install is an `LcdoctorError` (exit 2) with instructions.
- `--fix` only rewrites when the same name is importable at the new path. Semantic migrations (initialize_agent -> create_agent) are reported, never rewritten.
- Every bug found on a real repo gets a regression test named after what broke.

# Gotchas
- `langgraph` is a PEP 420 namespace package (no `__init__.py`); don't filter package dirs on `__init__.py`.
- `from pkg import sub` must fall back to importing the submodule, exactly like Python does.
- LangChain signals some moves with plain `UserWarning`s ("no longer supported"), not `DeprecationWarning`.
- langchain-community 0.4 depends on langchain-classic, so classic is often already installed after an upgrade.
