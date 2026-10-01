import json

from lcdoctor.diagnose import BROKEN, DEPRECATED, OK, Diagnosis, Finding, Location, Relocation
from lcdoctor.report import (
    brief,
    deprecation_sentence,
    next_steps,
    render_json,
    render_markdown,
    render_text,
)

DEP = {
    "since": "0.1.0",
    "removal": {"package": "langchain-classic", "version": "2.0.0"},
    "message": "Use `create_agent` instead. It will be removed in langchain-classic==2.0.0.",
    "source": "docstring",
}


def diag() -> Diagnosis:
    loc = [Location("agent.py", 3, "from langchain.agents import initialize_agent")]
    return Diagnosis(
        root="/proj",
        files_scanned=2,
        environment={
            "label": "fresh env",
            "python": "3.12.1",
            "executable": "/x",
            "distributions": {"langchain": "1.4.3", "langchain-core": "1.6.6"},
        },
        findings=[
            Finding(
                "langchain.agents",
                "initialize_agent",
                BROKEN,
                loc,
                detail="no attribute",
                relocation=Relocation(
                    "langchain_classic.agents",
                    "initialize_agent",
                    "langchain-classic",
                    "1.0.8",
                    False,
                    DEP,
                ),
            ),
            Finding(
                "langchain.vectorstores", "FAISS", BROKEN, loc, detail="No module", searched=True
            ),
            Finding("langgraph.prebuilt", "create_react_agent", DEPRECATED, loc, deprecation=DEP),
            Finding("langchain_core.tools", "tool", OK, loc),
        ],
        syntax_errors=[],
        mode="upgrade",
        locator_used=True,
    )


def test_deprecation_sentence_does_not_repeat_removal():
    s = deprecation_sentence(DEP)
    assert (
        s
        == "deprecated since 0.1.0, removal in langchain-classic==2.0.0. Use `create_agent` instead."
    )
    assert deprecation_sentence(None) == ""


def test_brief_flattens_rst_links_and_truncates():
    text = (
        "First sentence. Second one has `docs <https://x.dev/a>`_ in it. Third is dropped. "
        "Fourth too."
    )
    assert brief(text) == "First sentence. Second one has docs (https://x.dev/a) in it."
    assert brief("word " * 200).endswith("…")


def test_next_steps_counts_installs_fixes_and_migrations():
    steps = next_steps(diag())
    assert steps[0].startswith("pip install langchain-classic")
    assert any("--fix" in s and "1 import" in s for s in steps)
    assert any("still need a real migration" in s and s.startswith("2 imports") for s in steps)
    assert any("no new home" in s for s in steps)


def test_text_report_hides_ok_by_default():
    out = render_text(diag())
    assert "BROKEN       from langchain.agents import initialize_agent" in out
    assert (
        "now:  from langchain_classic.agents import initialize_agent   (langchain-classic 1.0.8, not installed)"
        in out
    )
    assert "not found in the latest LangChain-family releases" in out
    assert "langchain_core.tools" not in out
    assert "langchain_core.tools" in render_text(diag(), show_ok=True)
    assert "\033[" not in out


def test_markdown_report_table():
    md = render_markdown(diag())
    assert "**2 failing**, **1 deprecated**, 1 ok" in md
    assert "| BROKEN | `from langchain.agents import initialize_agent` | agent.py:3 |" in md
    assert "(needs `langchain-classic`)" in md


def test_json_report_is_complete():
    data = json.loads(render_json(diag()))
    assert data["summary"] == {
        "broken": 2,
        "not_installed": 0,
        "import_error": 0,
        "deprecated": 1,
        "ok": 1,
    }
    assert data["findings"][0]["relocation"]["module"] == "langchain_classic.agents"
    assert data["findings"][0]["import_line"] == "from langchain.agents import initialize_agent"
