"""
expense_crew.py
Expense Claim Crew: domain expert AI agents (CrewAI + Gemini) that audit employee
expense claims against company policy.
Author: Aman Aggarwal (065064), FORE School of Management

Design principle: the LLM agents read messy receipts, interpret findings and write
explanations; every calculation (limits, dates, duplicates) is done by deterministic
Python tools, so the agents never have to guess numbers.
"""
import json
import os
import re
import time
from datetime import date
from typing import List, Literal

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")   # no tracing prompt
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
# In CrewAI 1.9 this flag only switches off the interactive "view your traces? [y/N]"
# prompt, which would otherwise pause every run for 20 s (and hang a web app).
os.environ.setdefault("CREWAI_TESTING", "true")

import pandas as pd
from pydantic import BaseModel, Field
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool

# ------------------------------------------------------------------
# Compatibility fix: Gemini rejects requests whose last message is from the model
# ("400 Requests ending with a model turn are not supported"). CrewAI's tool loop
# can end on the agent's own "Observation" message, so we add a short user turn.
# ------------------------------------------------------------------
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "10"))   # stay safely under the free-tier limit
_last_call = [0.0]


def _wait_for_slot():
    gap = 60.0 / GEMINI_RPM
    wait = _last_call[0] + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def _patch_gemini_last_turn():
    try:
        from google.genai import types as gtypes
        from crewai.llms.providers.gemini.completion import GeminiCompletion
    except ImportError:
        return
    if getattr(GeminiCompletion, "_last_turn_patched", False):
        return
    original = GeminiCompletion._format_messages_for_gemini

    def patched(self, messages):
        contents, system_instruction = original(self, messages)
        if contents and contents[-1].role == "model":
            parts = contents[-1].parts or []
            if not any(getattr(p, "function_call", None) for p in parts):
                contents.append(gtypes.Content(
                    role="user", parts=[gtypes.Part.from_text(text="Continue.")]))
        return contents, system_instruction

    GeminiCompletion._format_messages_for_gemini = patched

    # Free-tier pacing: never send more than GEMINI_RPM requests per minute (all agents
    # together), and when Google answers 429 (quota) or 503 (busy), wait the time it asks
    # for and try the same call again instead of failing the whole claim.
    original_call = GeminiCompletion.call

    def paced_call(self, *args, **kwargs):
        for attempt in range(5):
            _wait_for_slot()
            try:
                return original_call(self, *args, **kwargs)
            except Exception as e:
                msg = str(e)
                if "PerDay" in msg or "per day" in msg.lower():
                    raise  # daily quota used up: waiting a minute will not help
                if not any(code in msg for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                    raise
                if attempt == 4:
                    raise
                m = re.search(r"retry in ([0-9.]+)s", msg)
                wait = float(m.group(1)) + 3 if m else 30
                print(f"   Gemini busy or rate-limited, waiting {wait:.0f} s then retrying ...")
                time.sleep(min(wait, 70))

    GeminiCompletion.call = paced_call
    GeminiCompletion._last_turn_patched = True


_patch_gemini_last_turn()

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ------------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------------
CLAIMS = pd.read_csv(os.path.join(DATA_DIR, "claims.csv"), dtype={"claim_id": str})
EMPLOYEES = pd.read_csv(os.path.join(DATA_DIR, "employees.csv"))
HISTORY = pd.read_csv(os.path.join(DATA_DIR, "claims_history.csv"))
with open(os.path.join(DATA_DIR, "expense_policy.md")) as f:
    POLICY_TEXT = f.read()

NO_RECEIPT = "NO RECEIPT ATTACHED"
TIER1 = {"Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune"}
LIMITS = {  # category: (unit, {grade: limit per unit})
    "Client Meals": ("person", {"G1": 1000, "G2": 1500, "G3": 2500}),
    "Local Travel": ("day", {"G1": 1000, "G2": 1500, "G3": 2500}),
    "Hotel Tier 1": ("night", {"G1": 4000, "G2": 6000, "G3": 9000}),
    "Hotel Tier 2": ("night", {"G1": 3000, "G2": 4500, "G3": 7000}),
    "Airfare": ("one-way trip", {"G1": 8000, "G2": 12000, "G3": 20000}),
    "Office Supplies": ("claim", {"G1": 2000, "G2": 2000, "G3": 2000}),
    "Internet and Mobile": ("month", {"G1": 1000, "G2": 1000, "G3": 1000}),
}
SUBMISSION_DAYS = 30
RECEIPT_THRESHOLD = 500


def _claim(claim_id: str) -> dict:
    row = CLAIMS[CLAIMS.claim_id == claim_id.strip().upper()]
    if row.empty:
        raise ValueError(f"Claim {claim_id} not found")
    return row.iloc[0].to_dict()


def _employee(employee_id: str) -> dict:
    return EMPLOYEES[EMPLOYEES.employee_id == employee_id].iloc[0].to_dict()


def _limit(claim: dict, grade: str):
    category = claim["category"]
    if category == "Hotel":
        category = "Hotel Tier 1" if claim["city"] in TIER1 else "Hotel Tier 2"
    unit, by_grade = LIMITS[category]
    per_unit = by_grade[grade]
    return category, unit, per_unit, per_unit * int(claim["units"])


# ------------------------------------------------------------------
# 2. Tools (deterministic Python the agents call)
# ------------------------------------------------------------------
@tool("Get claim details")
def get_claim_details(claim_id: str) -> str:
    """Return the expense claim form, the receipt text and the employee's grade
    for one claim ID such as C001."""
    c = _claim(claim_id)
    e = _employee(c["employee_id"])
    return json.dumps({**c, "employee_name": e["name"], "grade": e["grade"],
                       "department": e["department"]}, default=str)


@tool("Check policy rules")
def check_policy_rules(claim_id: str, receipt_total: float,
                       non_reimbursable_amount: float) -> str:
    """Apply the company expense policy (limits, receipts, timeliness, non-reimbursable
    items) to a claim. Pass the receipt total and the total of non-reimbursable items
    found on the receipt (0 if none). Returns triggered rules and the eligible amount."""
    c = _claim(claim_id)
    grade = _employee(c["employee_id"])["grade"]
    amount = float(c["amount"])
    category, unit, per_unit, max_allowed = _limit(c, grade)
    triggers = []

    has_receipt = c["receipt_text"].strip().upper() != NO_RECEIPT
    if not has_receipt and amount > RECEIPT_THRESHOLD:
        triggers.append({"rule": "R1 Receipts", "effect": "REJECT",
                         "detail": f"No receipt for a claim of INR {amount:,.0f} (above {RECEIPT_THRESHOLD})."})

    days = (date.fromisoformat(c["submitted_on"]) - date.fromisoformat(c["expense_date"])).days
    if days > SUBMISSION_DAYS:
        triggers.append({"rule": "R2 Timeliness", "effect": "REJECT",
                         "detail": f"Submitted {days} days after the expense (limit {SUBMISSION_DAYS})."})

    base = min(amount, float(receipt_total)) if has_receipt and receipt_total else amount
    eligible = max(base - float(non_reimbursable_amount or 0), 0.0)
    if non_reimbursable_amount and non_reimbursable_amount > 0:
        if eligible <= 0:
            triggers.append({"rule": "R3 Non-reimbursable", "effect": "REJECT",
                             "detail": "The whole claim is for non-reimbursable items."})
        else:
            triggers.append({"rule": "R3 Non-reimbursable", "effect": "FLAG",
                             "detail": f"INR {non_reimbursable_amount:,.0f} of non-reimbursable items removed."})

    if eligible > max_allowed:
        triggers.append({"rule": "R4 Limit", "effect": "FLAG",
                         "detail": f"Eligible INR {eligible:,.0f} is above the {category} limit of "
                                   f"INR {max_allowed:,.0f} ({c['units']} x {per_unit:,} per {unit}, grade {grade})."})

    return json.dumps({
        "claim_id": c["claim_id"], "category": category, "grade": grade,
        "limit_per_unit": per_unit, "unit": unit, "units": int(c["units"]),
        "max_allowed_amount": max_allowed, "claimed_amount": amount,
        "eligible_amount": round(eligible, 2),
        "recommended_amount": round(min(eligible, max_allowed), 2),
        "days_to_submit": days, "rule_triggers": triggers,
    })


@tool("Run fraud checks")
def run_fraud_checks(claim_id: str, receipt_date: str, receipt_total: float) -> str:
    """Check a claim for duplicates, split bills, weekend client meals, suspicious round
    amounts and mismatches between the claim and the receipt. Pass the receipt date as
    YYYY-MM-DD (empty if no receipt) and the receipt total (0 if no receipt)."""
    c = _claim(claim_id)
    grade = _employee(c["employee_id"])["grade"]
    amount = float(c["amount"])
    triggers = []

    same = lambda df: df[(df.employee_id == c["employee_id"]) & (df.vendor == c["vendor"])
                         & (df.expense_date == c["expense_date"])]
    dup_hist = same(HISTORY)[lambda d: d.amount == amount]
    dup_now = same(CLAIMS)[lambda d: (d.amount == amount) & (d.claim_id < c["claim_id"])]
    if len(dup_hist) or len(dup_now):
        ref = (list(dup_hist.claim_id) + list(dup_now.claim_id))[0]
        triggers.append({"rule": "R5 Duplicate", "effect": "REJECT",
                         "detail": f"Same employee, vendor, date and amount as claim {ref}."})

    siblings = same(CLAIMS)[lambda d: (d.claim_id != c["claim_id"]) & (d.amount != amount)]
    if len(siblings):
        _, _, _, max_allowed = _limit(c, grade)
        combined = amount + siblings.amount.sum()
        if combined > max_allowed:
            triggers.append({"rule": "R7 Split bill", "effect": "FLAG",
                             "detail": f"With {', '.join(siblings.claim_id)} the same-day total from "
                                       f"{c['vendor']} is INR {combined:,.0f}, above the limit of INR {max_allowed:,.0f}."})

    weekday = date.fromisoformat(c["expense_date"]).strftime("%A")
    if c["category"] == "Client Meals" and weekday in ("Saturday", "Sunday"):
        triggers.append({"rule": "R7 Weekend client meal", "effect": "FLAG",
                         "detail": f"Client meal on a {weekday}; business purpose needs confirmation."})

    if amount >= 5000 and amount % 1000 == 0:
        triggers.append({"rule": "R7 Round amount", "effect": "FLAG",
                         "detail": f"Round amount of INR {amount:,.0f}."})

    if receipt_total and abs(float(receipt_total) - amount) > 1:
        triggers.append({"rule": "R6 Receipt match", "effect": "FLAG",
                         "detail": f"Claimed INR {amount:,.0f} but the receipt shows INR {float(receipt_total):,.0f}."})
    if receipt_date and receipt_date != c["expense_date"]:
        triggers.append({"rule": "R6 Receipt match", "effect": "FLAG",
                         "detail": f"Claim date {c['expense_date']} but receipt date {receipt_date}."})

    return json.dumps({"claim_id": c["claim_id"], "expense_weekday": weekday,
                       "rule_triggers": triggers})


# ------------------------------------------------------------------
# 3. Structured outputs (what each agent must return)
# ------------------------------------------------------------------
class RuleTrigger(BaseModel):
    rule: str
    effect: Literal["REJECT", "FLAG"]
    detail: str


class NonReimbursableItem(BaseModel):
    item: str
    amount: float
    reason: str


class ReceiptExtraction(BaseModel):
    receipt_present: bool
    vendor: str = ""
    receipt_date: str = Field("", description="YYYY-MM-DD, empty if no receipt")
    receipt_total: float = Field(0, description="Final total on the receipt in INR, 0 if no receipt")
    non_reimbursable_items: List[NonReimbursableItem] = []
    notes: str = ""


class PolicyReport(BaseModel):
    rule_triggers: List[RuleTrigger] = []
    max_allowed_amount: float
    eligible_amount: float
    recommended_amount: float
    summary: str


class FraudReport(BaseModel):
    rule_triggers: List[RuleTrigger] = []
    risk_level: Literal["Low", "Medium", "High"]
    summary: str


class FinalDecision(BaseModel):
    decision: Literal["APPROVE", "FLAG", "REJECT"]
    recommended_amount: float
    reasons: List[str]
    employee_message: str


# ------------------------------------------------------------------
# 4. Agents and tasks
# ------------------------------------------------------------------
def make_llm(model: str = "gemini/gemini-flash-lite-latest", api_key: str | None = None) -> LLM:
    return LLM(model=model, api_key=api_key or os.getenv("GEMINI_API_KEY"), temperature=0.1)


def build_crew(llm, verbose: bool = False, max_rpm: int | None = None,
               task_callback=None) -> Crew:
    receipt_agent = Agent(
        role="Receipt Analyst",
        goal="Read the receipt for claim {claim_id} accurately and extract its facts",
        backstory=("You have processed thousands of Indian restaurant bills, cab receipts, hotel "
                   "invoices and e-tickets. You read messy receipt text, find the final total and "
                   "date, and spot items a company never pays for: alcohol (beer, wine, whisky, "
                   "spirits), traffic or parking fines, penalties, personal items, spa and salon. "
                   "You never guess: if something is not on the receipt, you say so."),
        tools=[get_claim_details], llm=llm, verbose=verbose, allow_delegation=False)

    policy_agent = Agent(
        role="Expense Policy Expert",
        goal="Decide which company policy rules apply to claim {claim_id}",
        backstory=("You are the company's expense policy owner in Finance. You always use the "
                   "'Check policy rules' tool for limits, dates and amounts, because policy "
                   "decisions must be exact. You know this policy by heart:\n\n" + POLICY_TEXT),
        tools=[check_policy_rules], llm=llm, verbose=verbose, allow_delegation=False)

    fraud_agent = Agent(
        role="Fraud and Risk Auditor",
        goal="Detect duplicate, split, suspicious or mismatched claims for claim {claim_id}",
        backstory=("You are an internal auditor trained in expense fraud patterns: duplicate "
                   "submissions, bills split to stay under limits, weekend entertainment and "
                   "receipts that do not match the claim. You rely on the 'Run fraud checks' tool "
                   "for evidence and never accuse without it."),
        tools=[run_fraud_checks], llm=llm, verbose=verbose, allow_delegation=False)

    approver_agent = Agent(
        role="Finance Approver",
        goal="Make a fair, policy-compliant final decision on claim {claim_id} and explain it",
        backstory=("You are the Finance Controller who signs off reimbursements. You combine the "
                   "receipt, policy and fraud findings, apply the decision rules exactly, and write "
                   "a short, polite message the employee will understand."),
        llm=llm, verbose=verbose, allow_delegation=False)

    receipt_task = Task(
        description=("Use the 'Get claim details' tool for claim {claim_id}. Read the receipt_text. "
                     "If it says 'NO RECEIPT ATTACHED', set receipt_present to false and the total to 0. "
                     "Otherwise extract the vendor, the receipt date as YYYY-MM-DD and the final total "
                     "in INR, and list every non-reimbursable item with its amount."),
        expected_output="The receipt facts in the required structure.",
        agent=receipt_agent, output_pydantic=ReceiptExtraction)

    policy_task = Task(
        description=("For claim {claim_id}, call the 'Check policy rules' tool with the receipt total "
                     "and the sum of non-reimbursable item amounts from the Receipt Analyst. Report "
                     "exactly the rule triggers and amounts the tool returns, then summarise in one "
                     "or two sentences."),
        expected_output="The policy findings in the required structure.",
        agent=policy_agent, context=[receipt_task], output_pydantic=PolicyReport)

    fraud_task = Task(
        description=("For claim {claim_id}, call the 'Run fraud checks' tool with the receipt date and "
                     "receipt total from the Receipt Analyst. Report exactly the rule triggers it returns. "
                     "Risk level: High if any REJECT trigger, Medium if any FLAG trigger, else Low."),
        expected_output="The fraud findings in the required structure.",
        agent=fraud_agent, context=[receipt_task], output_pydantic=FraudReport)

    decision_task = Task(
        description=("Make the final decision on claim {claim_id} using the policy and fraud findings.\n"
                     "Decision rules: REJECT if any trigger has effect REJECT (recommended amount 0). "
                     "Otherwise FLAG if any trigger has effect FLAG (recommended amount = the policy "
                     "report's recommended amount, or the receipt total if lower when the receipt does "
                     "not match). Otherwise APPROVE the policy report's recommended amount.\n"
                     "Give one reason per triggered rule (or 'All policy checks passed') and a two or "
                     "three sentence message to the employee."),
        expected_output="The final decision in the required structure.",
        agent=approver_agent, context=[receipt_task, policy_task, fraud_task],
        output_pydantic=FinalDecision)

    return Crew(agents=[receipt_agent, policy_agent, fraud_agent, approver_agent],
                tasks=[receipt_task, policy_task, fraud_task, decision_task],
                process=Process.sequential, verbose=verbose, max_rpm=max_rpm,
                tracing=False, task_callback=task_callback)


# ------------------------------------------------------------------
# 5. Running claims (with retry and model fallback)
# ------------------------------------------------------------------
# Flash-Lite first: on the free tier it is busy (503) far less often than Flash
FALLBACK_MODELS = ["gemini/gemini-flash-lite-latest", "gemini/gemini-flash-latest"]


def run_claim(claim_id: str, models=FALLBACK_MODELS, verbose: bool = False,
              api_key: str | None = None, retries: int = 2, task_callback=None) -> dict:
    """Run the four-agent crew on one claim. Returns each agent's structured output."""
    last_error = None
    for model in models:
        for attempt in range(retries):
            try:
                crew = build_crew(make_llm(model, api_key), verbose=verbose,
                                  task_callback=task_callback)
                start = time.time()
                result = crew.kickoff(inputs={"claim_id": claim_id})
                outs = [t.pydantic.model_dump() if t.pydantic else {"raw": t.raw}
                        for t in result.tasks_output]
                return {"claim_id": claim_id, "model": model,
                        "seconds": round(time.time() - start, 1),
                        "receipt": outs[0], "policy": outs[1], "fraud": outs[2], "decision": outs[3]}
            except Exception as e:  # busy model, rate limit or bad output
                last_error = e
                if "PerDay" in str(e) or "per day" in str(e).lower():
                    return {"claim_id": claim_id, "error": "DAILY QUOTA REACHED: " + str(e)[:300]}
                busy = any(s in str(e) for s in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
                time.sleep(20 if busy else 3)
    return {"claim_id": claim_id, "error": str(last_error)[:500]}


# ------------------------------------------------------------------
# 6. Adding a new claim (used by the Streamlit app)
# ------------------------------------------------------------------
CATEGORIES = ["Client Meals", "Local Travel", "Hotel", "Airfare", "Office Supplies",
              "Internet and Mobile"]


def add_claim(employee_id: str, expense_date: str, submitted_on: str, category: str,
              vendor: str, city: str, units: int, amount: float, receipt_text: str) -> str:
    """Add a new claim to the in-memory claims table and return its new ID."""
    global CLAIMS
    if employee_id not in set(EMPLOYEES.employee_id):
        raise ValueError("Unknown employee")
    if category not in CATEGORIES:
        raise ValueError("Unknown category")
    new_id = f"N{int(time.time() * 1000) % 10_000_000:07d}"
    row = {"claim_id": new_id, "employee_id": employee_id, "expense_date": expense_date,
           "submitted_on": submitted_on, "category": category, "vendor": vendor.strip(),
           "city": city.strip(), "units": int(units), "amount": float(amount),
           "receipt_text": receipt_text.strip() or NO_RECEIPT}
    CLAIMS = pd.concat([CLAIMS, pd.DataFrame([row])], ignore_index=True)
    return new_id
