"""Streamlit chat UI for YojanaMitra AI.   Run:  streamlit run Home.py"""
import re
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import build_agent, text_of
from config import OUTPUT_DIR

st.set_page_config(page_title="YojanaMitra AI", page_icon="🇮🇳", layout="centered")

EXAMPLES = {
    "Student (Maharashtra)": "I'm a 21 year old female student from a village in Maharashtra, OBC, "
                             "family income 2.4 lakh. My father owns 2 acres of farmland and we "
                             "don't have an LPG connection. What can my family get?",
    "Small business": "I'm a 28 year old woman from rural Pune district, OBC, family income "
                      "3 lakh. I want to start a tailoring unit costing 8 lakh. Any schemes?",
    "Senior citizen": "My grandfather is 72, lives in Nashik city, family income 6 lakh, "
                      "general category. Is there any health scheme for him?",
}


@st.cache_resource
def get_agent():
    return build_agent()


def file_links(text: str) -> list[Path]:
    return [OUTPUT_DIR / name for name in re.findall(r"outputs/([\w.\-]+)", text)]


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []          # list of dicts for display

with st.sidebar:
    st.header("YojanaMitra AI")
    st.caption("Finds government schemes you qualify for, using RAG over official scheme "
               "documents and a LangGraph agent with 5 tools.")
    st.subheader("Try an example")
    for label, prompt in EXAMPLES.items():
        if st.button(label, width="stretch"):
            st.session_state.pending = prompt
    st.divider()
    if st.button("New conversation", width="stretch"):
        st.session_state.clear()
        st.rerun()
    st.caption("Covers 8 central and Maharashtra schemes. Always verify on myscheme.gov.in.")

st.title("🇮🇳 YojanaMitra AI")
st.write("Tell me your age, income, state, category and occupation, and I'll find the "
         "schemes you qualify for.")

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        if item.get("steps"):
            with st.expander(f"Agent used {len(item['steps'])} tool call(s)"):
                for s in item["steps"]:
                    st.markdown(f"**{s['name']}** `{s['args']}`")
                    st.code(s["result"][:1500], language="text")
        st.markdown(item["content"])
        for path in item.get("files", []):
            if path.exists():
                st.download_button(f"Download {path.name}", path.read_bytes(),
                                   file_name=path.name, key=f"{path.name}-{id(item)}")

prompt = st.chat_input("Describe your situation...") or st.session_state.pop("pending", None)
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        steps, files = [], []
        with st.status("Finding schemes...", expanded=True) as status:
            config = {"configurable": {"thread_id": st.session_state.thread_id},
                      "recursion_limit": 25}
            final = None
            try:
                for update in get_agent().stream({"messages": [HumanMessage(prompt)]},
                                                 config, stream_mode="updates"):
                    for node, data in update.items():
                        for msg in data["messages"]:
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    st.write(f"🔧 Calling **{tc['name']}**")
                                    steps.append({"id": tc["id"], "name": tc["name"],
                                                  "args": tc["args"], "result": ""})
                            elif isinstance(msg, ToolMessage):
                                for s in steps:
                                    if s["id"] == msg.tool_call_id:
                                        s["result"] = text_of(msg.content)
                                files += file_links(text_of(msg.content))
                            elif isinstance(msg, AIMessage):
                                final = text_of(msg.content)
                status.update(label=f"Done - {len(steps)} tool call(s)", state="complete",
                              expanded=False)
            except Exception as exc:  # show the error instead of a blank page
                status.update(label="Error", state="error")
                final = f"Something went wrong: `{exc}`. Check your API key in `.env`."
        answer = final or "I couldn't produce an answer. Please try rephrasing."
        st.markdown(answer)
        for path in files:
            if path.exists():
                st.download_button(f"Download {path.name}", path.read_bytes(),
                                   file_name=path.name, key=f"new-{path.name}")
    st.session_state.history.append({"role": "assistant", "content": answer,
                                     "steps": steps, "files": files})
