"""
Support-ticket triage with a WorkflowGraph and a ValidationTool quality gate.

Shows:
- ``WorkflowGraph`` nodes as plain functions, with conditional edges (first match wins).
- A ``ValidationTool`` gate: refund replies must quote an order id, or the ticket is escalated.
- ``metadata["nodes_visited"]`` to audit which path each ticket took.

Needs: nothing beyond ``pip install agentensemble``. No API key, no network.

Run: python examples/support_ticket_triage.py
"""

import re
from typing import Any

from agentensemble import ValidationTool, WorkflowGraph

TICKETS = [
    "Hi, I want a refund for order #48213, the blender arrived broken.",
    "I'd like my money back please, the shoes don't fit.",
    "How do I reset my password? The link expired.",
    "Your app crashes every time I open the camera. This is urgent!",
]

REPLIES = {
    "refund": "We have opened a refund for order {order_id}. Expect it within 14 days.",
    "account": "Use 'Forgot password' on the sign-in page; reset links expire after 60 minutes.",
    "bug": "Thanks for the report. Our engineers will look into it.",
}

refund_gate = ValidationTool(required_pattern=r"#\d{5}", reject_pattern=r"\{order_id\}|#None")


def classify(state: dict[str, Any]) -> dict[str, Any]:
    text = state["query"].lower()
    if any(w in text for w in ("refund", "money back", "return")):
        category = "refund"
    elif any(w in text for w in ("password", "login", "sign in")):
        category = "account"
    else:
        category = "bug"
    order = re.search(r"#(\d{5})", state["query"])
    urgent = "urgent" in text or "crash" in text
    return {
        "context": {"category": category, "order_id": f"#{order[1]}" if order else None},
        "priority": "high" if urgent else "normal",
    }


def draft_reply(state: dict[str, Any]) -> dict[str, Any]:
    ctx = state["context"]
    return {"result": REPLIES[ctx["category"]].format(order_id=ctx["order_id"])}


def quality_check(state: dict[str, Any]) -> dict[str, Any]:
    if state["context"]["category"] != "refund":
        return {"checked": True, "passed": True}
    verdict = refund_gate.run(state["result"])
    return {"checked": True, "passed": verdict["valid"], "why": verdict["justifications"]}


def escalate(state: dict[str, Any]) -> dict[str, Any]:
    reason = "; ".join(state.get("why") or ["high priority"])
    return {"result": f"ESCALATED to a human agent ({reason})"}


def build_graph() -> WorkflowGraph:
    graph = WorkflowGraph(entry="classify")
    graph.add_node("classify", classify)
    graph.add_node("draft_reply", draft_reply)
    graph.add_node("quality_check", quality_check)
    graph.add_node("escalate", escalate)
    graph.add_edge("classify", "escalate", lambda s: s["priority"] == "high")
    graph.add_edge("classify", "draft_reply")
    graph.add_edge("draft_reply", "quality_check")
    graph.add_edge("quality_check", "escalate", lambda s: not s["passed"])
    graph.add_edge("quality_check", "end")
    graph.add_edge("escalate", "end")
    return graph


def main() -> None:
    graph = build_graph()
    for ticket in TICKETS:
        out = graph.run(ticket)
        path = " -> ".join(out["metadata"]["nodes_visited"])
        print(f"Ticket: {ticket}\n  path:  {path}\n  reply: {out['result']}\n")


if __name__ == "__main__":
    main()
