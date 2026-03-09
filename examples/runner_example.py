"""
Runner Example - Central Entry Point

Demonstrates Runner.run(), Runner.arun(), RunConfig, and RunHooks.
Uses StateGraphAgent (no API key required).
"""

from agentensemble import Runner, RunConfig, RunHooks, StateGraphAgent


def main():
    # Simple graph: start -> end
    def start_node(state):
        return {"context": {**state.context, "started": True}, "result": f"Processed: {state.query}"}

    agent = StateGraphAgent(
        name="demo",
        nodes={"start": start_node},
        max_iterations=5,
    )
    agent._route = lambda state, current: "end"  # start -> done

    # Basic run
    result = Runner.run(agent, "Hello world")
    print("Result:", result["result"])

    # With RunConfig and hooks
    started = []
    ended = []

    def on_start(query, kwargs):
        started.append(query)
        print(f"  [hook] on_start: {query}")

    def on_end(res):
        ended.append(res)
        print(f"  [hook] on_end: {res.get('result', '')[:50]}...")

    config = RunConfig(
        hooks=RunHooks(on_start=on_start, on_end=on_end),
        context={"extra": "value"},
    )
    result = Runner.run(agent, "With hooks", config=config)
    print("Result:", result["result"])
    assert len(started) == 1 and len(ended) == 1


if __name__ == "__main__":
    main()
