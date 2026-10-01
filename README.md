# lcdoctor

Find the LangChain imports that break when you upgrade, and where each one lives now.

```
$ lcdoctor upgrade examples/support-agent

BROKEN       from langchain.agents import initialize_agent
             agent.py:3
             why:  module 'langchain.agents' has no attribute 'initialize_agent'
             now:  from langchain_classic.agents import initialize_agent   (langchain-classic 1.0.8, installed)
             but:  deprecated since 0.1.0, removal in langchain-classic==2.0.0. Use
               `langchain.agents.create_agent` for new applications.

BROKEN       from langchain.prompts import PromptTemplate
             chains.py:4
             why:  No module named 'langchain.prompts'
             now:  from langchain_core.prompts import PromptTemplate   (langchain-core 1.6.6, installed)

next
  lcdoctor upgrade examples/support-agent --fix    # rewrites 11 imports to their new home
  5 imports still need a real migration; see the 'deprecated' lines
```

## Why this exists

Agents that worked in the demo break the day someone bumps `langchain`. LangChain 1.0 moved
`initialize_agent`, `AgentExecutor`, `LLMChain`, the memory classes and dozens of loaders out of
`langchain` and into `langchain-classic`, `langchain-community`, `langchain-core` and the partner
packages. On 1.x the old import fails with a bare `AttributeError` or `ModuleNotFoundError`,
and the error doesn't say where the name went.

LangChain's own packages already have the answer. Removed names still exist somewhere, and deprecated ones
carry their since version, removal version and replacement in a PEP 702 `__deprecated__`
attribute and in their docstrings. lcdoctor reads that directly, so there's no hand-written
migration table to fall out of date.

## Install

```bash
uv tool install git+https://github.com/reetbatra/lcdoctor
# or
pipx install git+https://github.com/reetbatra/lcdoctor
```

Needs Python 3.11+. `upgrade` also needs [uv](https://docs.astral.sh/uv/) on your PATH.

## Use

```bash
lcdoctor upgrade .                          # what breaks if you upgrade to the latest releases?
lcdoctor upgrade . --to langchain==1.4.3    # or to a specific release
lcdoctor upgrade . --diff                   # show the import rewrites
lcdoctor upgrade . --fix                    # apply them

lcdoctor check .                            # are the imports OK in the project's .venv today?
lcdoctor check . --python path/to/python    # or in any interpreter you point it at

lcdoctor check . --format markdown          # for a CI job summary or a PR comment
lcdoctor check . --format json              # for anything else
```

Exit codes: `0` nothing failing, `1` failing imports (or deprecations with `--strict`),
`2` lcdoctor itself couldn't run. That makes `lcdoctor upgrade . --strict` usable as a CI gate.

## How it works

1. Every `.py` file is parsed with `ast` and each `langchain*` / `langgraph*` import
   is collected. Your code never runs. Packages that live inside your project are skipped
   (Langchain-Chatchat ships its own `langchain_chatchat`, which isn't a dependency).
2. A small stdlib-only script runs inside the target interpreter. It imports each
   name the way your code does and records three things: whether it resolves, whether it's
   deprecated (from `__deprecated__`, the docstring, or a warning raised on import), and
   which module actually defines it.
   - `check` probes your project's interpreter.
   - `upgrade` builds a throwaway env with uv. It installs the latest release of every
     LangChain package you import or list in `requirements*.txt` / `pyproject.toml`, so you
     see the break before you upgrade.
3. For each broken name, lcdoctor searches a static index of everything the latest
   LangChain-family packages export (`__all__`, LangChain's lazy-import tables, top-level
   definitions). It probes the best candidates and picks the home in the package that defines
   the object, preferring paths that aren't deprecated themselves.
4. Optionally, `--fix` rewrites `from X import Y` statements to point at the new home.
   It only makes mechanical moves, where the same object is available at a new path. Moving
   from `initialize_agent` to `create_agent` changes behaviour, so lcdoctor reports it and
   leaves the change to you.

Environments and indexes are cached in `~/.cache/lcdoctor` (override with `LCDOCTOR_CACHE`).
"Latest" is re-resolved once a day, or on demand with `--refresh`.

## Tried on real projects

`lcdoctor upgrade` against LangChain 1.4.3, on public commits you can check out and re-run:

| Project | Python files | Imports that break | New home found | Still broken after `--fix` |
|---|---|---|---|---|
| [chatchat-space/Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) @ `49165d6` | 281 | 66 | 61 | 6 |
| [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag) @ `cec956e` | 81 | 10 | 10 | 0 |
| [1Panel-dev/MaxKB](https://github.com/1Panel-dev/MaxKB) @ `2c7c8c9` | 1,057 | 0 | n/a | n/a |

The six left in Langchain-Chatchat need real migrations. Five use `langchain_core.pydantic_v1`,
which 1.0 removed in favour of plain `pydantic`. The sixth is `NextStepOutput`, an internal type
alias with no public home. MaxKB is already on 1.x, which checks that lcdoctor stays quiet when
nothing is wrong.

Running it on these repos also turned up three bugs in lcdoctor, each now covered by a
regression test:
- `langgraph` is a namespace package with no `__init__.py`, and the index was skipping it.
- A project's own `langchain_*` package was being treated as a dependency.
- `from langchain import hub` names a submodule, not an attribute.

## Limits

- It checks imports, not calls. A removed method like `chain.run()`, on an object that still
  imports fine, won't show up.
- The locator indexes `langchain`, `langchain-core`, `langchain-classic`,
  `langchain-community`, `langchain-text-splitters`, `langgraph`, `langchain-openai` and
  `langchain-anthropic`, plus every LangChain package your project imports or declares. A name
  that moved into another partner package is reported as having no new home.
- `--fix` doesn't rewrite bare `import langchain.x` statements. They're still reported.
- Type aliases defined by plain assignment (like `NextStepOutput`) aren't indexed.

## Develop

```bash
uv sync
uv run pytest                    # unit tests, no network
uv run pytest -m integration     # real envs from PyPI, about a minute
uv run ruff check src tests && uv run mypy
```

`examples/support-agent` is a small docs-support agent written against LangChain 0.3, for trying
the tool end to end.

## License

MIT
