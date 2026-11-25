"""Evaluation dashboard: runs the evaluation cases live through the real tools."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from tools import evaluate_eligibility, hybrid_search  # noqa: E402

st.set_page_config(page_title="Evaluation · YojanaMitra AI", page_icon="📊", layout="wide")
st.title("📊 Evaluation")
st.write("Computed live from `eval/eval_cases.json` with the same rule engine and hybrid "
         "retriever the agent uses. For the end-to-end agent score run "
         "`python eval/evaluate.py --agent`.")

cases = json.loads((ROOT / "eval" / "eval_cases.json").read_text(encoding="utf-8"))


@st.cache_data
def run_eligibility():
    rows = []
    for c in cases["eligibility"]:
        r = evaluate_eligibility(**c["input"])
        got_e, got_l = sorted(x["id"] for x in r["eligible"]), sorted(x["id"] for x in r["likely"])
        ok = got_e == sorted(c["eligible"]) and got_l == sorted(c["likely"])
        rows.append({"Profile": c["name"],
                     "Expected eligible": ", ".join(sorted(c["eligible"])) or "none",
                     "Expected likely": ", ".join(sorted(c["likely"])) or "none",
                     "Engine eligible": ", ".join(got_e) or "none",
                     "Engine likely": ", ".join(got_l) or "none",
                     "Result": "✅ Pass" if ok else "❌ Fail"})
    return rows


@st.cache_data
def run_retrieval(k=4):
    rows = []
    for c in cases["retrieval"]:
        sources = [d.metadata["source"] for d in hybrid_search(c["query"], k=k)]
        rank = sources.index(c["source"]) + 1 if c["source"] in sources else None
        rows.append({"Question": c["query"], "Expected document": c["source"],
                     "Rank": rank, "Top results": ", ".join(sources)})
    return rows


elig, ret = run_eligibility(), run_retrieval()
passed = sum(r["Result"].startswith("✅") for r in elig)
hits = sum(r["Rank"] is not None for r in ret)
mrr = sum(1 / r["Rank"] for r in ret if r["Rank"]) / len(ret)

c1, c2, c3 = st.columns(3)
c1.metric("Eligibility exact match", f"{passed}/{len(elig)}")
c2.metric("Retrieval Hit@4", f"{hits}/{len(ret)}")
c3.metric("Mean reciprocal rank", f"{mrr:.2f}")

st.subheader("Eligibility test cases")
st.dataframe(elig, width="stretch", hide_index=True)

st.subheader("Retrieval test questions")
st.bar_chart({r["Question"][:40] + "…": (1 / r["Rank"] if r["Rank"] else 0) for r in ret},
             horizontal=True, x_label="Reciprocal rank (1.0 = expected document ranked first)")
st.dataframe(ret, width="stretch", hide_index=True)
