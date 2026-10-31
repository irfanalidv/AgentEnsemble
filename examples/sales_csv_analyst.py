"""
CSV analytics agent: answer business questions about a sales file with typed tools.

Shows:
- Exposing deterministic Python analytics as ``@function_tool`` tools, so the numbers come
  from code and the LLM only chooses the tool and phrases the answer.
- ``ReActAgent`` picking between several tools from the question.
- ``metadata`` with iteration and tool-call counts.

Needs: nothing beyond ``pip install agentensemble``; reads ``examples/data/sales.csv``. No API
key: runs offline with a rule-based stand-in LLM. With ``OPENAI_API_KEY`` (or
``MISTRAL_API_KEY``) set, a real model chooses the tools and writes the answer.

Run: python examples/sales_csv_analyst.py [--offline]
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

from offline_llm import RuleBasedLLM, choose_llm

from agentensemble import ReActAgent, function_tool

SALES_CSV = Path(__file__).parent / "data" / "sales.csv"
GROUPS = ("region", "product", "month")


def load_rows() -> list[dict[str, str]]:
    with SALES_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


@function_tool(description="Total revenue (units * unit_price) grouped by region, product or month")
def revenue_by(group: str) -> str:
    if group not in GROUPS:
        return f"Unknown group {group!r}; use one of {GROUPS}"
    totals: dict[str, float] = defaultdict(float)
    for row in load_rows():
        key = row["date"][:7] if group == "month" else row[group]
        totals[key] += int(row["units"]) * float(row["unit_price"])
    ranked = dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))
    return json.dumps({"group": group, "revenue_usd": ranked})


@function_tool(description="Total units sold for one product name")
def units_sold(product: str) -> str:
    units = sum(int(r["units"]) for r in load_rows() if r["product"].lower() == product.lower())
    return json.dumps({"product": product, "units": units})


def plan(question: str, tools):
    q = question.lower()
    for name in ("widget", "gadget", "gizmo"):
        if name in q and "units" in q:
            return "units_sold", {"product": name.title()}
    for group in GROUPS:
        if group in q:
            return "revenue_by", {"group": group}
    return None


def answer(question: str, results: list[str]) -> str:
    if not results:
        return "I can answer questions about revenue by region, product or month."
    data = json.loads(results[-1])
    if "units" in data:
        return f"{data['product']}: {data['units']} units sold."
    ranked = data["revenue_usd"]
    top, value = next(iter(ranked.items()))
    breakdown = ", ".join(f"{k} ${v:,.2f}" for k, v in ranked.items())
    return f"Top {data['group']} is {top} with ${value:,.2f} (all: {breakdown})."


def main() -> None:
    agent = ReActAgent(
        name="analyst", tools=[revenue_by, units_sold], llm=choose_llm(RuleBasedLLM(plan, answer))
    )
    for question in (
        "Which region had the highest revenue?",
        "How did revenue compare by month?",
        "How many units of Widget did we sell?",
    ):
        out = agent.run(question)
        print(f"Q: {question}\nA: {out['result']}")
        print(f"   ({out['metadata']['tool_calls']} tool call(s))\n")


if __name__ == "__main__":
    main()
