"""The five tools the YojanaMitra agent can call.

1. search_schemes           - RAG: hybrid (FAISS + BM25) search over official scheme text
2. check_eligibility        - exact rule checks against data/schemes.json
3. calculate_benefit        - money maths the LLM should never guess
4. generate_checklist_pdf   - creates a downloadable PDF
5. create_deadline_reminder - creates a downloadable calendar (.ics) file
"""
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

from fpdf import FPDF
from langchain_core.tools import tool

from config import OUTPUT_DIR, SCHEMES_JSON, VECTORSTORE_DIR, get_embeddings

with open(SCHEMES_JSON, encoding="utf-8") as f:
    SCHEMES: dict[str, dict] = {s["id"]: s for s in json.load(f)}

VALID_FLAGS = {"owns_farmland", "no_lpg_connection", "daughter_under_10",
               "wants_to_start_business", "no_pucca_house"}
SPECIAL_CATEGORIES = {"sc", "st", "obc", "minority"}


def inr(n: int) -> str:
    """Indian digit grouping: 280000 -> 2,80,000."""
    s = str(int(n))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    return ",".join(([head] if head else []) + parts) + "," + tail


# ---------------------------------------------------------------- RAG search
@lru_cache(maxsize=1)
def _retrievers():
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS

    if not (VECTORSTORE_DIR / "index.faiss").exists():
        raise RuntimeError("Vector store not found. Run `python ingest.py` first.")
    db = FAISS.load_local(str(VECTORSTORE_DIR), get_embeddings(),
                          allow_dangerous_deserialization=True)
    all_docs = list(db.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(all_docs, k=6)
    return db, bm25


def hybrid_search(query: str, k: int = 4):
    """Reciprocal-rank fusion of semantic (FAISS) and keyword (BM25) results."""
    db, bm25 = _retrievers()
    ranked_lists = [db.similarity_search(query, k=6), bm25.invoke(query)]
    scores, docs = {}, {}
    for ranked in ranked_lists:
        for rank, d in enumerate(ranked):
            key = d.page_content
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            docs[key] = d
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [docs[key] for key in best]


@tool
def search_schemes(query: str) -> str:
    """Search the official text of government schemes (benefits, eligibility,
    exclusions, documents, how to apply, helplines). Use it to get details and
    to back every claim with a source. Returns passages tagged with their source file."""
    hits = hybrid_search(query)
    return "\n\n---\n\n".join(f"[Source: {d.metadata['source']}]\n{d.page_content}" for d in hits)


# ------------------------------------------------------- eligibility rules
def _norm_gender(g: str) -> str:
    g = (g or "").strip().lower()
    if g in {"f", "female", "woman", "women", "girl", "lady"}:
        return "female"
    if g in {"m", "male", "man", "boy"}:
        return "male"
    return g


def _check_route(route: dict, p: dict) -> tuple[bool, str]:
    """Return (passes, reason_if_not)."""
    if "min_age" in route and p["age"] < route["min_age"]:
        return False, f"minimum age {route['min_age']}"
    if "max_age" in route and p["age"] > route["max_age"]:
        return False, f"maximum age {route['max_age']}"
    if "max_income" in route and p["income"] > route["max_income"]:
        return False, f"family income must be at most Rs {inr(route['max_income'])}"
    if "screening_max_income" in route and p["income"] > route["screening_max_income"]:
        return False, "income too high for this low-income route"
    if "genders" in route and p["gender"] not in route["genders"]:
        return False, f"only for {', '.join(route['genders'])} applicants"
    if "categories" in route and p["category"] not in route["categories"]:
        return False, f"only for category {', '.join(route['categories']).upper()}"
    if "occupations" in route and p["occupation"] not in route["occupations"]:
        return False, f"only for {', '.join(route['occupations'])}s"
    if "states" in route and p["state"] not in route["states"]:
        return False, f"only for residents of {', '.join(s.title() for s in route['states'])}"
    if "areas" in route and p["area"] not in route["areas"]:
        return False, f"only for {', '.join(route['areas'])} areas"
    missing = [r for r in route.get("requires", []) if r not in p["flags"]]
    if missing:
        return False, "needs: " + ", ".join(m.replace("_", " ") for m in missing)
    return True, ""


def evaluate_eligibility(age, gender, annual_family_income, category="general",
                         occupation="other", state="", area="urban", flags=None) -> dict:
    """Pure-Python rule engine (also used by the evaluation script)."""
    p = {
        "age": int(age), "gender": _norm_gender(gender), "income": int(annual_family_income),
        "category": (category or "general").strip().lower(),
        "occupation": (occupation or "other").strip().lower(),
        "state": (state or "").strip().lower(), "area": (area or "urban").strip().lower(),
        "flags": {f.strip().lower() for f in (flags or [])} & VALID_FLAGS,
    }
    result = {"eligible": [], "likely": [], "not_eligible": []}
    for s in SCHEMES.values():
        reasons = []
        for route in s["routes"]:
            ok, why = _check_route(route, p)
            if ok:
                bucket = "likely" if "screening_max_income" in route else "eligible"
                result[bucket].append({"id": s["id"], "name": s["name"], "route": route["label"],
                                       "benefit": s["benefit_note"], "also_check": s["also_check"]})
                break
            reasons.append(why)
        else:
            result["not_eligible"].append({"id": s["id"], "name": s["name"], "reason": reasons[0]})
    # a scheme found via a definite route should not also sit in "likely"
    eligible_ids = {e["id"] for e in result["eligible"]}
    result["likely"] = [x for x in result["likely"] if x["id"] not in eligible_ids]
    return result


@tool
def check_eligibility(age: int, gender: str, annual_family_income: int,
                      category: str = "general", occupation: str = "other",
                      state: str = "", area: str = "urban",
                      flags: Optional[list[str]] = None) -> str:
    """Check which schemes a person qualifies for, using exact rules.

    Args:
        age: age in years of the person applying.
        gender: male or female.
        annual_family_income: total yearly family income in rupees (2.5 lakh = 250000).
        category: general, ews, obc, sc, st or minority.
        occupation: student, farmer, business, salaried, unemployed, homemaker or other.
        state: state of residence in lowercase, e.g. maharashtra.
        area: rural or urban.
        flags: any that apply: owns_farmland (family owns farm land),
            no_lpg_connection, daughter_under_10, wants_to_start_business, no_pucca_house.
    """
    r = evaluate_eligibility(age, gender, annual_family_income, category,
                             occupation, state, area, flags)
    lines = ["ELIGIBLE:"]
    lines += [f"- {e['id']}: {e['name']} (via {e['route']}). Benefit: {e['benefit']}. "
              f"Also confirm: {e['also_check']}" for e in r["eligible"]] or ["- none"]
    lines.append("LIKELY ELIGIBLE (needs list/document verification):")
    lines += [f"- {e['id']}: {e['name']} (via {e['route']}). Also confirm: {e['also_check']}"
              for e in r["likely"]] or ["- none"]
    lines.append("NOT ELIGIBLE:")
    lines += [f"- {e['id']}: {e['name']} ({e['reason']})" for e in r["not_eligible"]]
    return "\n".join(lines)


# ---------------------------------------------------------- benefit maths
def pmegp_subsidy(project_cost: int, category: str, gender: str, area: str,
                  sector: str) -> tuple[int, float, int]:
    cap = 5_000_000 if sector == "manufacturing" else 2_000_000
    special = category.lower() in SPECIAL_CATEGORIES or _norm_gender(gender) == "female"
    rate = {(False, "urban"): 0.15, (False, "rural"): 0.25,
            (True, "urban"): 0.25, (True, "rural"): 0.35}[(special, area.lower())]
    eligible_cost = min(project_cost, cap)
    return round(eligible_cost * rate), rate, eligible_cost


@tool
def calculate_benefit(scheme_ids: list[str], category: str = "general", gender: str = "",
                      area: str = "urban", project_cost: int = 0,
                      sector: str = "manufacturing", hilly_state: bool = False) -> str:
    """Calculate money benefits for a list of scheme ids (from check_eligibility).

    Args:
        scheme_ids: e.g. ["pm_kisan", "ladki_bahin"].
        category, gender, area: needed for the PMEGP subsidy rate.
        project_cost: planned project cost in rupees (PMEGP only).
        sector: manufacturing or service (PMEGP only).
        hilly_state: True for NE / Himalayan states (PMAY-G only).
    """
    yearly, one_time, lines = 0, 0, []
    for sid in scheme_ids:
        s = SCHEMES.get(sid)
        if not s:
            lines.append(f"- {sid}: unknown scheme id")
            continue
        if sid == "pmegp":
            if project_cost <= 0:
                lines.append("- PMEGP: subsidy is 15-35% of project cost; give project_cost to compute it")
                continue
            amt, rate, cost = pmegp_subsidy(project_cost, category, gender, area, sector)
            one_time += amt
            lines.append(f"- {s['name']}: one-time subsidy Rs {inr(amt)} "
                         f"({rate:.0%} of eligible project cost Rs {inr(cost)})")
        elif sid == "pmay_g":
            amt = 130_000 if hilly_state else 120_000
            one_time += amt
            lines.append(f"- {s['name']}: one-time Rs {inr(amt)} for house construction (+ up to Rs 12,000 for a toilet)")
        elif s["cash_yearly"]:
            yearly += s["cash_yearly"]
            lines.append(f"- {s['name']}: Rs {inr(s['cash_yearly'])} per year ({s['benefit_note']})")
        else:
            lines.append(f"- {s['name']}: non-cash benefit ({s['benefit_note']})")
    lines.append(f"TOTAL recurring cash: Rs {inr(yearly)} per year")
    if one_time:
        lines.append(f"TOTAL one-time support: Rs {inr(one_time)}")
    return "\n".join(lines)


# ---------------------------------------------------------------- PDF
def _latin(text: str) -> str:
    """Built-in PDF fonts only support Latin-1."""
    text = text.replace("₹", "Rs ").replace("–", "-").replace("—", "-").replace("’", "'")
    return text.encode("latin-1", "replace").decode("latin-1")


@tool
def generate_checklist_pdf(person_name: str, scheme_ids: list[str]) -> str:
    """Create a PDF checklist with the benefit, documents to collect and how to
    apply for each scheme. Returns the saved file path."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, _latin("YojanaMitra AI - Scheme Checklist"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, _latin(f"Prepared for: {person_name}    Date: {datetime.now():%d %b %Y}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    count = 0
    for sid in scheme_ids:
        s = SCHEMES.get(sid)
        if not s:
            continue
        count += 1
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(230, 240, 255)
        pdf.multi_cell(0, 8, _latin(f"{count}. {s['name']}"), fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10.5)
        pdf.multi_cell(0, 6, _latin(f"Benefit: {s['benefit_note']}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.cell(0, 6, "Documents to collect:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10.5)
        for doc in s["documents"]:
            y = pdf.get_y()
            pdf.rect(pdf.l_margin + 2, y + 1.5, 3.5, 3.5)
            pdf.set_x(pdf.l_margin + 8)
            pdf.multi_cell(0, 6, _latin(doc), new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 6, _latin(f"How to apply: {s['apply']}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(110, 110, 110)
        pdf.multi_cell(0, 5.5, _latin(f"Also confirm: {s['also_check']}"), new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 5.5, _latin(f"Source: {s['source']}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
    if count == 0:
        return "No valid scheme ids given; no PDF created."
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, _latin("Generated by YojanaMitra AI. Rules change - verify on the "
                                "official portal before applying."), new_x="LMARGIN", new_y="NEXT")
    safe = re.sub(r"[^A-Za-z0-9]+", "_", person_name).strip("_")[:30] or "user"
    path = OUTPUT_DIR / f"checklist_{safe}_{uuid.uuid4().hex[:6]}.pdf"
    pdf.output(str(path))
    return f"Checklist PDF created with {count} schemes: outputs/{path.name}"


# ---------------------------------------------------------------- reminder
@tool
def create_deadline_reminder(scheme_id: str, date: str, note: str = "") -> str:
    """Create a calendar reminder (.ics file, opens in Google Calendar / Outlook)
    to apply for a scheme on a date. date must be YYYY-MM-DD. Adds an alert
    one day before."""
    s = SCHEMES.get(scheme_id)
    if not s:
        return f"Unknown scheme id '{scheme_id}'."
    try:
        day = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return "Date must be in YYYY-MM-DD format."
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    desc = f"Apply via {s['apply']}. Documents: {', '.join(s['documents'])}. {note}".strip()
    desc = desc.replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;")

    def fold(line: str) -> str:  # iCalendar lines must be <= 75 characters
        return "\r\n ".join(line[i:i + 73] for i in range(0, len(line), 73)) or line

    ics = "\r\n".join(fold(line) for line in [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//YojanaMitra AI//EN", "CALSCALE:GREGORIAN",
        "BEGIN:VEVENT", f"UID:{uuid.uuid4()}@yojanamitra", f"DTSTAMP:{stamp}",
        f"DTSTART;VALUE=DATE:{day:%Y%m%d}",
        f"DTEND;VALUE=DATE:{day + timedelta(days=1):%Y%m%d}",
        f"SUMMARY:Apply for {s['name']}", f"DESCRIPTION:{desc}",
        "BEGIN:VALARM", "ACTION:DISPLAY", "TRIGGER:-P1D",
        f"DESCRIPTION:Tomorrow: apply for {s['name']}", "END:VALARM",
        "END:VEVENT", "END:VCALENDAR", ""])
    path = OUTPUT_DIR / f"reminder_{scheme_id}_{day:%Y%m%d}.ics"
    path.write_text(ics, encoding="utf-8")
    return f"Calendar reminder created for {day:%d %b %Y}: outputs/{path.name}"


TOOLS = [search_schemes, check_eligibility, calculate_benefit,
         generate_checklist_pdf, create_deadline_reminder]
