"""Scheme Explorer: browse the knowledge base the agent's tools use."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from tools import SCHEMES  # noqa: E402

st.set_page_config(page_title="Scheme Explorer · YojanaMitra AI", page_icon="📚", layout="wide")
st.title("📚 Scheme Explorer")
st.write("The 8 schemes in the knowledge base: the same records the eligibility and benefit "
         "tools read. Always confirm on the official portal.")

query = st.text_input("Filter by name or keyword", placeholder="e.g. woman, farmer, scholarship")
level = st.radio("Level", ["All", "Central", "Maharashtra"], horizontal=True)

shown = [s for s in SCHEMES.values()
         if (level == "All" or s["level"] == level)
         and (not query or query.lower() in (s["name"] + s["benefit_note"] + s["also_check"]
                                             + " ".join(r["label"] for r in s["routes"])).lower())]
st.caption(f"Showing {len(shown)} of {len(SCHEMES)} schemes")

cols = st.columns(2)
for i, s in enumerate(shown):
    with cols[i % 2].container(border=True):
        st.caption(s["level"])
        st.subheader(s["name"])
        st.write(s["benefit_note"])
        st.markdown("**Who qualifies:** " + "; or ".join(r["label"] for r in s["routes"]))
        st.markdown(f"**Also check:** {s['also_check']}")
        with st.expander("Documents and how to apply"):
            st.markdown("\n".join(f"- {d}" for d in s["documents"]))
            st.write(s["apply"])
        st.markdown(f"[Official source]({s['source']})")
