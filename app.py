"""
Expense Claim Crew: Streamlit front end
Four domain expert AI agents (CrewAI + Gemini) audit an expense claim against company policy.
Author: Aman Aggarwal (065064), FORE School of Management
Run locally:  streamlit run app.py
"""
import json
import os
from datetime import date

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# The Gemini key comes from .env on the Mac, or from Streamlit secrets when deployed
load_dotenv()
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

import expense_crew as ec  # noqa: E402  (needs the key in the environment first)

st.set_page_config(page_title="Expense Claim Crew", page_icon="🧾", layout="wide")

AGENTS = [
    ("Receipt Analyst", "Reads the receipt: vendor, date, total, items the company never pays for"),
    ("Expense Policy Expert", "Applies grade-wise limits, receipt and 30-day submission rules"),
    ("Fraud and Risk Auditor", "Looks for duplicates, split bills, weekend meals, receipt mismatches"),
    ("Finance Approver", "Combines all findings into a decision and writes to the employee"),
]
DECISION_STYLE = {
    "APPROVE": ("#1F6F4A", "#E7F3EC", "Approved"),
    "FLAG": ("#9A5B00", "#FDF1DC", "Flagged for manager review"),
    "REJECT": ("#A12A22", "#FBE9E7", "Rejected"),
}

if "runs" not in st.session_state:
    st.session_state.runs = {}          # claim_id -> crew result (so a claim is not re-run)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def employee_label(eid):
    e = ec.EMPLOYEES[ec.EMPLOYEES.employee_id == eid].iloc[0]
    return f"{e['name']} ({eid}, grade {e['grade']}, {e['department']})"


def show_claim(claim):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Amount claimed", f"INR {claim['amount']:,.0f}")
    c2.metric("Category", claim["category"])
    c3.metric("Expense date", claim["expense_date"])
    c4.metric("Submitted on", claim["submitted_on"])
    st.write(f"**Employee:** {employee_label(claim['employee_id'])}  \n"
             f"**Vendor:** {claim['vendor']}, {claim['city']}  \n"
             f"**Units:** {claim['units']} (attendees for meals, nights for hotels, days for travel)")
    st.markdown("**Receipt text** (as read from the scanned receipt)")
    st.code(claim["receipt_text"].replace(" | ", "\n"), language=None)


def run_with_progress(claim_id):
    """Run the crew, ticking off each agent as its task finishes."""
    box = st.status("The crew is reviewing this claim ...", expanded=True)
    lines = [box.empty() for _ in AGENTS]
    for line, (name, _) in zip(lines, AGENTS):
        line.markdown(f"⏳ {name}")
    done = {"n": 0}

    def on_task_done(_output):
        i = done["n"]
        if i < len(lines):
            lines[i].markdown(f"✅ {AGENTS[i][0]}")
        done["n"] += 1

    with box:
        result = ec.run_claim(claim_id, task_callback=on_task_done)
    if "error" in result:
        box.update(label="The crew could not finish", state="error")
    else:
        box.update(label=f"Review finished in {result['seconds']} s ({result['model']})",
                   state="complete", expanded=False)
    st.session_state.runs[claim_id] = result
    return result


def show_result(result):
    if "error" in result:
        msg = result["error"]
        if msg.startswith("DAILY QUOTA"):
            st.error("The free daily Gemini quota is used up. Please try again tomorrow. "
                     "The Results tab still shows the full evaluation.")
        else:
            st.error("Gemini is busy or unavailable right now. Please try again in a minute.")
            with st.expander("Technical details"):
                st.code(msg)
        return

    d = result["decision"]
    color, bg, label = DECISION_STYLE[d["decision"]]
    st.markdown(
        f"""<div style="border-left:6px solid {color};background:{bg};padding:1rem 1.3rem;
        border-radius:6px;margin:0.5rem 0 1rem 0">
        <div style="color:{color};font-weight:600">Final decision</div>
        <div style="color:{color};font-size:1.8rem;font-weight:700;line-height:1.3">
        {label}: INR {d['recommended_amount']:,.0f}</div></div>""",
        unsafe_allow_html=True)

    st.markdown("**Reasons**")
    for r in d["reasons"]:
        st.markdown(f"- {r}")
    st.markdown("**Message to the employee**")
    st.info(d["employee_message"])

    st.markdown("**What each agent found**")
    receipt, policy, fraud = result["receipt"], result["policy"], result["fraud"]
    with st.expander("1. Receipt Analyst"):
        st.write(f"Receipt attached: **{'Yes' if receipt.get('receipt_present') else 'No'}**  \n"
                 f"Vendor: {receipt.get('vendor') or 'n/a'}  \n"
                 f"Receipt date: {receipt.get('receipt_date') or 'n/a'}  \n"
                 f"Receipt total: INR {receipt.get('receipt_total', 0):,.0f}")
        items = receipt.get("non_reimbursable_items") or []
        if items:
            st.write("Non-reimbursable items found:")
            st.dataframe(pd.DataFrame(items), hide_index=True, width="stretch")
        else:
            st.write("No non-reimbursable items found.")
    with st.expander("2. Expense Policy Expert"):
        st.write(f"Policy limit for this claim: INR {policy.get('max_allowed_amount', 0):,.0f}  \n"
                 f"Eligible amount: INR {policy.get('eligible_amount', 0):,.0f}")
        triggers = policy.get("rule_triggers") or []
        if triggers:
            st.dataframe(pd.DataFrame(triggers), hide_index=True, width="stretch")
        else:
            st.write("No policy rules triggered.")
        st.caption(policy.get("summary", ""))
    with st.expander("3. Fraud and Risk Auditor"):
        st.write(f"Risk level: **{fraud.get('risk_level', 'n/a')}**")
        triggers = fraud.get("rule_triggers") or []
        if triggers:
            st.dataframe(pd.DataFrame(triggers), hide_index=True, width="stretch")
        else:
            st.write("No fraud signals found.")
        st.caption(fraud.get("summary", ""))


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Expense Claim Crew")
    st.write("Four domain expert AI agents audit an employee expense claim against the "
             "company policy and recommend Approve, Flag or Reject.")
    for i, (name, desc) in enumerate(AGENTS, 1):
        st.markdown(f"**{i}. {name}**  \n<span style='font-size:0.85rem'>{desc}</span>",
                    unsafe_allow_html=True)
    st.divider()
    st.write("**Design:** the agents read receipts and explain decisions; every limit, date "
             "and duplicate check is done by Python tools, so no agent guesses a number.")
    if not os.getenv("GEMINI_API_KEY"):
        st.error("No Gemini API key found. Add it to .env (Mac) or the app's Secrets (cloud).")
    st.caption("AI: Google Gemini (free tier) via CrewAI. Company and data are fictional. "
               "Built by Aman Aggarwal (065064), FORE School of Management.")

st.title("Expense Claim Crew")
st.write("AI agents that check expense claims for **Nimbus Analytics Pvt. Ltd.** (a fictional "
         "company). Each review takes about a minute, because the free Gemini tier allows only a "
         "few requests per minute.")

tab_sample, tab_new, tab_results, tab_policy = st.tabs(
    ["Review a sample claim", "Submit a new claim", "Results on all 16 claims", "Company policy"])

# ------------------------------------------------------------------
# Tab 1: sample claims
# ------------------------------------------------------------------
with tab_sample:
    expected = pd.read_csv(os.path.join(ec.DATA_DIR, "expected_outcomes.csv")).set_index("claim_id")
    samples = ec.CLAIMS[ec.CLAIMS.claim_id.str.startswith("C")]
    options = {f"{r.claim_id}: {r.category}, {r.vendor}, INR {r.amount:,.0f}": r.claim_id
               for r in samples.itertuples()}
    choice = st.selectbox("Choose a claim", list(options), index=3)
    cid = options[choice]
    claim = ec.CLAIMS[ec.CLAIMS.claim_id == cid].iloc[0].to_dict()
    show_claim(claim)

    with st.expander("What should the answer be? (answer key)"):
        e = expected.loc[cid]
        st.write(f"**{e.expected_decision}**, INR {e.expected_amount:,.0f}: {e.reason}")

    if st.button("Run the crew on this claim", type="primary", key="run_sample"):
        run_with_progress(cid)
    if cid in st.session_state.runs:
        show_result(st.session_state.runs[cid])

# ------------------------------------------------------------------
# Tab 2: new claim
# ------------------------------------------------------------------
with tab_new:
    st.write("Enter a claim as an employee would. Paste the receipt text (or leave it empty "
             "to test a claim without a receipt).")
    with st.form("new_claim"):
        c1, c2 = st.columns(2)
        emp = c1.selectbox("Employee", list(ec.EMPLOYEES.employee_id), format_func=employee_label)
        category = c2.selectbox("Category", ec.CATEGORIES)
        vendor = c1.text_input("Vendor", "Spice Route Restaurant")
        city = c2.text_input("City", "Delhi")
        expense_date = c1.date_input("Expense date", date(2026, 10, 5))
        submitted_on = c2.date_input("Submitted on", date(2026, 10, 8))
        units = c1.number_input("Units (attendees, nights or days; 1 otherwise)", 1, 50, 2)
        amount = c2.number_input("Amount claimed (INR)", 0.0, 1_000_000.0, 2400.0, step=100.0)
        receipt = st.text_area(
            "Receipt text", height=150,
            value=("SPICE ROUTE RESTAURANT | New Delhi | Date: 05/10/2026 | Covers: 2 | "
                   "Paneer Tikka 420 | Dal Makhani 380 | Naan x4 240 | Red Wine x2 900 | "
                   "Sub-total 1940 | GST 97 | Service 363 | TOTAL INR 2400"))
        submitted = st.form_submit_button("Submit and review", type="primary")

    if submitted:
        problems = []
        if not vendor.strip():
            problems.append("Vendor is required.")
        if amount <= 0:
            problems.append("Amount must be more than zero.")
        if submitted_on < expense_date:
            problems.append("Submission date cannot be before the expense date.")
        if problems:
            for p in problems:
                st.error(p)
        else:
            new_id = ec.add_claim(emp, expense_date.isoformat(), submitted_on.isoformat(),
                                  category, vendor, city, int(units), float(amount), receipt)
            st.session_state.last_new = new_id
            st.success(f"Claim {new_id} created. The crew is reviewing it.")
            run_with_progress(new_id)

    last = st.session_state.get("last_new")
    if last and last in st.session_state.runs:
        st.caption(f"Result for claim {last}")
        show_result(st.session_state.runs[last])

# ------------------------------------------------------------------
# Tab 3: saved evaluation results (no API calls needed)
# ------------------------------------------------------------------
with tab_results:
    path = os.path.join("results", "evaluation.csv")
    if not os.path.exists(path):
        st.info("Run the notebook 01_expense_claim_crew.ipynb first to create results/evaluation.csv.")
    else:
        ev = pd.read_csv(path)
        c1, c2, c3 = st.columns(3)
        c1.metric("Decision accuracy", f"{ev.decision_correct.mean():.0%}",
                  f"{ev.decision_correct.sum()} of {len(ev)} claims", delta_color="off")
        c2.metric("Amount accuracy", f"{ev.amount_correct.mean():.0%}",
                  f"{ev.amount_correct.sum()} of {len(ev)} claims", delta_color="off")
        c3.metric("Average time per claim", f"{ev.seconds.mean():.0f} s")
        st.markdown("**Expected decision vs the crew's decision**")
        st.dataframe(pd.crosstab(ev.expected_decision, ev.crew_decision,
                                 rownames=["Expected"], colnames=["Crew decided"]),
                     width="content")
        st.markdown("**Every claim**")
        st.dataframe(ev[["claim_id", "expected_decision", "crew_decision", "expected_amount",
                         "crew_amount", "decision_correct", "reason"]],
                     hide_index=True, width="stretch")
        st.caption("These results come from the notebook run with Gemini Flash-Lite. The 16 test "
                   "claims were written together with the policy, so they show the system works "
                   "as designed, not how it would perform on real company data.")

# ------------------------------------------------------------------
# Tab 4: policy
# ------------------------------------------------------------------
with tab_policy:
    st.markdown(ec.POLICY_TEXT)
