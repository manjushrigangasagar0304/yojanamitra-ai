"""Tests that run without an API key:  python -m pytest -q

The agent test drives the real LangGraph graph and real tools with a scripted
chat model, proving the tool-calling loop works end to end.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from config import OUTPUT_DIR  # noqa: E402
from tools import (calculate_benefit, check_eligibility, create_deadline_reminder,  # noqa: E402
                   evaluate_eligibility, generate_checklist_pdf, pmegp_subsidy, search_schemes)


def test_eligibility_rules():
    r = evaluate_eligibility(21, "F", 240000, "obc", "student", "Maharashtra", "rural",
                             ["owns_farmland"])
    assert {x["id"] for x in r["eligible"]} == {"pm_kisan", "ladki_bahin"}
    text = check_eligibility.invoke({"age": 20, "gender": "female", "annual_family_income": 240000,
                                     "state": "maharashtra"})
    assert "ladki_bahin" in text and "minimum age 21" in text


def test_benefit_maths():
    assert pmegp_subsidy(800000, "obc", "female", "rural", "service") == (280000, 0.35, 800000)
    assert pmegp_subsidy(3000000, "general", "male", "urban", "service")[0] == 300000  # capped at 20L
    out = calculate_benefit.invoke({"scheme_ids": ["pm_kisan", "ladki_bahin"]})
    assert "Rs 24,000 per year" in out


def test_rag_search():
    out = search_schemes.invoke({"query": "Ladki Bahin monthly amount"})
    assert out.count("[Source:") == 4


def test_files_created():
    pdf = generate_checklist_pdf.invoke({"person_name": "Test User", "scheme_ids": ["pm_kisan", "pmuy"]})
    assert (OUTPUT_DIR / pdf.split("outputs/")[1]).read_bytes()[:4] == b"%PDF"
    ics = create_deadline_reminder.invoke({"scheme_id": "pms_sc", "date": "2026-12-15"})
    body = (OUTPUT_DIR / ics.split("outputs/")[1]).read_text()
    assert "BEGIN:VEVENT" in body and "20261215" in body


class ScriptedModel(BaseChatModel):
    """Returns pre-written AI messages in order, like an LLM that calls tools."""
    script: list
    step: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script[self.step]
        self.step += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_agent_tool_loop():
    from agent import build_agent
    profile = {"age": 21, "gender": "female", "annual_family_income": 240000,
               "category": "obc", "occupation": "student", "state": "maharashtra",
               "area": "rural", "flags": ["owns_farmland"]}
    script = [
        AIMessage("", tool_calls=[{"name": "check_eligibility", "args": profile, "id": "c1"}]),
        AIMessage("", tool_calls=[
            {"name": "search_schemes", "args": {"query": "Ladki Bahin documents"}, "id": "c2"},
            {"name": "calculate_benefit", "args": {"scheme_ids": ["pm_kisan", "ladki_bahin"]}, "id": "c3"}]),
        AIMessage("You qualify for PM-KISAN and Ladki Bahin: Rs 24,000 a year."),
    ]
    agent = build_agent(llm=ScriptedModel(script=script))
    out = agent.invoke({"messages": [("user", "what can I get?")]},
                       {"configurable": {"thread_id": "t"}})
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_msgs] == ["check_eligibility", "search_schemes", "calculate_benefit"]
    assert "Rs 24,000 per year" in tool_msgs[2].content
    assert out["messages"][-1].content.startswith("You qualify")
