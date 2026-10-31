"""
Order assistant with human approval for refunds, persistent memory, and tracing.

Shows:
- ``@function_tool`` tools with JSON schemas inferred from type hints.
- ``Runner`` + ``RunConfig``: a ``SQLiteSession`` for multi-turn memory, ``TraceHooks`` for
  observability, and ``interrupt_before_tools=["issue_refund"]`` for human-in-the-loop.
- Pausing before a sensitive tool, then resuming with the human's decision.

Needs: nothing beyond ``pip install agentensemble``. No API key: runs offline with a rule-based
stand-in LLM. With ``OPENAI_API_KEY`` (or ``MISTRAL_API_KEY``) set, a real model chooses the
tools. The session database is written to a temporary directory.

Run: python examples/order_assistant_approval.py [--offline]
"""

import re
import tempfile
from pathlib import Path

from offline_llm import RuleBasedLLM, choose_llm

from agentensemble import ReActAgent, RunConfig, Runner, SQLiteSession, TraceHooks, function_tool

ORDERS = {
    "48213": {"item": "blender", "status": "delivered", "total": 89.0},
    "51907": {"item": "headphones", "status": "in transit", "total": 149.0},
}


@function_tool(description="Look up an order by its 5-digit id")
def lookup_order(order_id: str) -> str:
    order = ORDERS.get(order_id.lstrip("#"))
    if not order:
        return f"No order {order_id}"
    return f"Order {order_id}: {order['item']}, {order['status']}, total ${order['total']:.2f}"


@function_tool(description="Issue a full refund for an order (requires human approval)")
def issue_refund(order_id: str) -> str:
    return f"Refund issued for order {order_id}"


def plan(question: str, tools):
    order = re.search(r"\d{5}", question)
    if not order:
        return None
    if "refund" in question.lower():
        return "issue_refund", {"order_id": order[0]}
    return "lookup_order", {"order_id": order[0]}


def answer(question: str, results: list[str]) -> str:
    return results[-1] if results else "Which order number do you mean?"


def print_event(event) -> None:
    if event.type.value in ("tool_start", "tool_end", "llm_end"):
        extra = event.data.get("tool") or event.data.get("usage") or ""
        took = f" {event.duration_ms:.1f} ms" if event.duration_ms is not None else ""
        print(f"    [trace] {event.type.value} {extra}{took}")


def main() -> None:
    llm = choose_llm(RuleBasedLLM(plan, answer))
    agent = ReActAgent(name="orders", tools=[lookup_order, issue_refund], llm=llm)

    with tempfile.TemporaryDirectory() as tmp:
        session = SQLiteSession("customer-42", db_path=str(Path(tmp) / "sessions.db"))
        config = RunConfig(
            session=session,
            trace_hooks=TraceHooks(on_event=print_event),
            interrupt_before_tools=["issue_refund"],
        )

        q1 = "Where is order 51907?"
        print(f"User: {q1}")
        print(f"Bot:  {Runner.run(agent, q1, config=config)['result']}\n")

        q2 = "Order 48213 arrived broken, please refund it."
        print(f"User: {q2}")
        paused = Runner.run(agent, q2, config=config)
        if paused.get("__interrupted__"):
            pending = paused["pending_tool"]
            print(f"  ! approval needed: {pending['name']}({pending['arguments']})")
            decision = issue_refund.run(**pending["arguments"])  # a human approved it
            print("  ! approved by supervisor")
            final = Runner.run(
                agent, q2, config=config, resume=paused["run_state"], resume_value=decision
            )
            print(f"Bot:  {final['result']}\n")

        roles = [m["role"] for m in session.get_messages()]
        print(f"Session now holds {len(roles)} messages: {roles}")


if __name__ == "__main__":
    main()
