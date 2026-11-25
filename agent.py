"""YojanaMitra agent: an explicit LangGraph ReAct loop.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END

The LLM decides which tool to call; LangGraph runs it and feeds the result
back until the LLM answers without asking for a tool.
"""
from datetime import date

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOLS

SYSTEM_PROMPT = f"""You are YojanaMitra, an assistant that helps Indian citizens find
government schemes they are entitled to. Today is {date.today():%d %B %Y}.

How to work:
1. You need age, gender, annual family income, category (general/ews/obc/sc/st/minority),
   occupation, state and rural/urban area. If any of these are missing, ask for them in ONE
   short message before calling tools. Also ask about land, LPG, house, daughters under 10
   and business plans only if the user's situation suggests they matter.
2. Call check_eligibility first. Then call search_schemes for the eligible schemes to get
   details, documents and how to apply.
3. Use calculate_benefit for every rupee figure. Never do money maths yourself.
4. When the user wants it, call generate_checklist_pdf and create_deadline_reminder.
5. Answer ONLY from tool results. Cite sources as [source: file.md]. If the knowledge base
   does not cover something, say so and point to myscheme.gov.in.
6. Clearly separate "Eligible" from "Likely eligible - needs verification".
7. Reply in the user's language (English, Hindi or Marathi), in short sections with bullets.
   End with the total benefit and the single most useful next step."""


def build_agent(llm=None, checkpointer=None):
    if llm is None:
        from config import get_llm
        llm = get_llm()
    model = llm.bind_tools(TOOLS)

    def call_model(state: MessagesState):
        response = model.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def text_of(content) -> str:
    """Some models return a list of content parts instead of a string."""
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content or ""


if __name__ == "__main__":
    agent = build_agent()
    query = ("I'm a 21 year old female student from a village in Maharashtra, OBC, "
             "family income 2.4 lakh. My father owns 2 acres of farmland and we don't have "
             "an LPG connection. What schemes can my family get? Make me a checklist PDF.")
    config = {"configurable": {"thread_id": "cli-demo"}, "recursion_limit": 25}
    for step in agent.stream({"messages": [("user", query)]}, config, stream_mode="values"):
        step["messages"][-1].pretty_print()
