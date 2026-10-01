from pathlib import Path

from lcdoctor.diagnose import BROKEN, DEPRECATED, Diagnosis, Finding, Location, Relocation
from lcdoctor.fix import apply_fixes, moves_from, plan_fixes, rewrite_source

MOVES = {
    ("langchain.agents", "initialize_agent"): "langchain_classic.agents",
    ("langchain.agents", "Tool"): "langchain_core.tools",
    ("langchain.prompts", "PromptTemplate"): "langchain_core.prompts",
}


def test_splits_one_statement_into_one_per_new_home():
    src = "from langchain.agents import AgentType, Tool, initialize_agent\n"
    out, moved = rewrite_source(src, MOVES)
    assert out == (
        "from langchain_classic.agents import initialize_agent\n"
        "from langchain.agents import AgentType\n"
        "from langchain_core.tools import Tool\n"
    ) or out == (
        "from langchain.agents import AgentType\n"
        "from langchain_core.tools import Tool\n"
        "from langchain_classic.agents import initialize_agent\n"
    )
    assert sorted(moved) == [
        ("langchain.agents.Tool", "langchain_core.tools.Tool"),
        ("langchain.agents.initialize_agent", "langchain_classic.agents.initialize_agent"),
    ]


def test_keeps_aliases_indentation_and_trailing_comments():
    src = (
        "def f():\n"
        "    from langchain.prompts import PromptTemplate as PT  # noqa: F401\n"
        "    return PT\n"
    )
    out, _ = rewrite_source(src, MOVES)
    assert out == (
        "def f():\n"
        "    from langchain_core.prompts import PromptTemplate as PT  # noqa: F401\n"
        "    return PT\n"
    )


def test_rewrites_parenthesized_multiline_imports():
    src = "from langchain.agents import (\n    Tool,\n    initialize_agent,\n)\nx = 1\n"
    out, _ = rewrite_source(src, MOVES)
    assert "from langchain_core.tools import Tool\n" in out
    assert "from langchain_classic.agents import initialize_agent\n" in out
    assert out.endswith("x = 1\n")
    assert "(" not in out


def test_leaves_shared_lines_and_unrelated_imports_alone():
    src = "import os; from langchain.prompts import PromptTemplate\nfrom langchain_openai import ChatOpenAI\n"
    out, moved = rewrite_source(src, MOVES)
    assert out == src and moved == []


def test_preserves_crlf():
    src = "from langchain.prompts import PromptTemplate\r\nx = 1\r\n"
    out, _ = rewrite_source(src, MOVES)
    assert out == "from langchain_core.prompts import PromptTemplate\r\nx = 1\r\n"


def _diag(root: Path, findings: list[Finding]) -> Diagnosis:
    return Diagnosis(str(root), 1, {}, findings, [], "upgrade", True)


def test_only_broken_findings_with_same_name_homes_become_moves(tmp_path: Path):
    loc = [Location("app.py", 1, "")]
    findings = [
        Finding(
            "langchain.prompts",
            "PromptTemplate",
            BROKEN,
            loc,
            relocation=Relocation(
                "langchain_core.prompts", "PromptTemplate", None, None, True, None
            ),
        ),
        Finding(
            "langchain.chains",
            None,
            BROKEN,
            loc,
            relocation=Relocation("langchain_classic.chains", None, None, None, True, None),
        ),
        Finding("langgraph.prebuilt", "create_react_agent", DEPRECATED, loc),
        Finding("langchain.memory", "Gone", BROKEN, loc),
    ]
    assert moves_from(_diag(tmp_path, findings)) == {
        ("langchain.prompts", "PromptTemplate"): "langchain_core.prompts",
    }


def test_plan_and_apply(tmp_path: Path):
    app = tmp_path / "app.py"
    app.write_text("from langchain.prompts import PromptTemplate\n")
    finding = Finding(
        "langchain.prompts",
        "PromptTemplate",
        BROKEN,
        [Location("app.py", 1, "")],
        relocation=Relocation("langchain_core.prompts", "PromptTemplate", None, None, True, None),
    )
    edits = plan_fixes(_diag(tmp_path, [finding]))
    assert len(edits) == 1
    assert "+from langchain_core.prompts import PromptTemplate" in edits[0].diff("app.py")
    assert (
        app.read_text() == "from langchain.prompts import PromptTemplate\n"
    )  # planning never writes
    apply_fixes(edits)
    assert app.read_text() == "from langchain_core.prompts import PromptTemplate\n"
