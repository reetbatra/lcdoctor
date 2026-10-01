"""End to end against real LangChain releases. Slow (builds envs with uv) and needs network:

uv run pytest -m integration
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lcdoctor import cli

pytestmark = pytest.mark.integration

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "support-agent"


def test_upgrade_finds_every_v1_break_then_fix_makes_it_import(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    project = tmp_path / "support-agent"
    shutil.copytree(EXAMPLE, project)

    assert cli.main(["upgrade", str(project), "--to", "langchain==1.4.3", "--format", "json"]) == 1
    report = json.loads(capsys.readouterr().out)
    broken = {f["target"]: f for f in report["findings"] if f["status"] == "broken"}
    assert len(broken) == 11
    homes = {t: f["relocation"]["module"] for t, f in broken.items()}
    assert homes["langchain.prompts.PromptTemplate"] == "langchain_core.prompts"
    assert homes["langchain.chat_models.ChatOpenAI"] == "langchain_openai"
    assert homes["langchain.agents.Tool"] == "langchain_core.tools"
    assert (
        homes["langchain.text_splitter.RecursiveCharacterTextSplitter"]
        == "langchain_text_splitters"
    )
    assert homes["langchain.agents.initialize_agent"] == "langchain_classic.agents"
    assert (
        broken["langchain.agents.initialize_agent"]["relocation"]["deprecation"]["removal"][
            "version"
        ]
        == "2.0.0"
    )

    assert cli.main(["upgrade", str(project), "--to", "langchain==1.4.3", "--fix"]) == 1
    capsys.readouterr()
    assert cli.main(["upgrade", str(project), "--to", "langchain==1.4.3", "--format", "json"]) == 0
    after = json.loads(capsys.readouterr().out)
    assert after["summary"]["broken"] == 0

    # The rewritten project really imports on the upgraded stack.
    env_python = after["environment"]["executable"]
    proc = subprocess.run(
        [env_python, "-W", "ignore", "-c", "import agent, chains, retriever, graph"],
        cwd=project,
        capture_output=True,
        text=True,
        env={"OPENAI_API_KEY": "sk-test-not-real", "PATH": ""},
    )
    assert proc.returncode == 0, proc.stderr


def test_check_against_a_real_v03_environment(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from lcdoctor.envs import build_env

    env = build_env(
        [
            "langchain==0.3.30",
            "langchain-community==0.3.31",
            "langchain-openai==0.3.35",
            "langgraph==0.6.9",
        ],
        f"{sys.version_info[0]}.{sys.version_info[1]}",
        purpose="test-v03",
    )
    code = cli.main(["check", str(EXAMPLE), "--python", str(env.python), "--format", "json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["summary"]["broken"] == 0
    deprecated = {f["target"]: f for f in report["findings"] if f["status"] == "deprecated"}
    assert deprecated["langchain.chains.LLMChain"]["deprecation"]["since"] == "0.1.17"
    assert (
        deprecated["langchain.chat_models.ChatOpenAI"]["deprecation"]["removal"]["package"]
        == "langchain-community"
    )
