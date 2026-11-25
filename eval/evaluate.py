"""Evaluate YojanaMitra.

    python eval/evaluate.py            # eligibility accuracy + retrieval hit rate (no API key needed)
    python eval/evaluate.py --agent    # also run every persona through the full agent (uses the LLM)

Metrics
- Eligibility exact-match: the rule engine returns exactly the expected eligible / likely sets.
- Retrieval Hit@4 and MRR: the expected scheme file appears in the top-4 hybrid search results.
- Agent scheme recall (optional): share of expected schemes the agent's final answer mentions.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import SCHEMES, evaluate_eligibility, hybrid_search  # noqa: E402

ALIASES = {"pm_kisan": ["pm-kisan", "pm kisan", "kisan samman"], "pmjay": ["pm-jay", "ayushman"],
           "pmuy": ["ujjwala"], "pmegp": ["pmegp", "employment generation"],
           "pms_sc": ["post-matric", "post matric"], "ssy": ["sukanya"],
           "ladki_bahin": ["ladki bahin"], "pmay_g": ["pmay-g", "awas"]}
assert set(ALIASES) == set(SCHEMES)

CASES = json.loads((ROOT / "eval" / "eval_cases.json").read_text(encoding="utf-8"))


def eval_eligibility():
    passed = 0
    print("\n== Eligibility rule engine ==")
    for case in CASES["eligibility"]:
        r = evaluate_eligibility(**case["input"])
        got_e = sorted(x["id"] for x in r["eligible"])
        got_l = sorted(x["id"] for x in r["likely"])
        ok = got_e == sorted(case["eligible"]) and got_l == sorted(case["likely"])
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {case['name']}")
        if not ok:
            print(f"      expected eligible={sorted(case['eligible'])} likely={sorted(case['likely'])}")
            print(f"      got      eligible={got_e} likely={got_l}")
    n = len(CASES["eligibility"])
    print(f"Exact-match accuracy: {passed}/{n} = {passed / n:.0%}")
    return passed / n


def eval_retrieval(k=4):
    hits, rr = 0, 0.0
    print("\n== Retrieval (hybrid FAISS + BM25) ==")
    for case in CASES["retrieval"]:
        sources = [d.metadata["source"] for d in hybrid_search(case["query"], k=k)]
        rank = sources.index(case["source"]) + 1 if case["source"] in sources else None
        hits += rank is not None
        rr += 1 / rank if rank else 0
        print(f"{'HIT ' if rank else 'MISS'} @{rank or '-'}  {case['query']}")
    n = len(CASES["retrieval"])
    print(f"Hit@{k}: {hits}/{n} = {hits / n:.0%}   MRR: {rr / n:.2f}")
    return hits / n, rr / n


def eval_agent():
    from agent import build_agent, text_of
    agent = build_agent()
    print("\n== End-to-end agent ==")
    recalls = []
    for i, case in enumerate(CASES["eligibility"]):
        p = case["input"]
        q = (f"I am {p['age']} years old, {p['gender']}, {p['category']} category, "
             f"{p['occupation']}, living in {p['area']} {p['state']}. Family income is "
             f"Rs {p['annual_family_income']}. Extra details: {', '.join(p['flags']) or 'none'}. "
             "Which schemes can I get? Do not create any files.")
        out = agent.invoke({"messages": [("user", q)]},
                           {"configurable": {"thread_id": f"eval-{i}"}, "recursion_limit": 25})
        answer = text_of(out["messages"][-1].content).lower()
        expected = case["eligible"] + case["likely"]
        found = [sid for sid in expected if any(a in answer for a in ALIASES[sid])]
        recall = len(found) / len(expected) if expected else 1.0
        recalls.append(recall)
        print(f"{recall:4.0%}  {case['name']}")
    print(f"Mean scheme recall: {sum(recalls) / len(recalls):.0%}")


if __name__ == "__main__":
    eval_eligibility()
    eval_retrieval()
    if "--agent" in sys.argv:
        eval_agent()
