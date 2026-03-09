# AgentEnsemble Framework Improvements

**Principal AI Infrastructure Engineer Review — 2025**

---

## 1. Codebase Audit Summary

### Architectural Weaknesses Identified

| Issue | Location | Status |
|-------|----------|--------|
| StructuredAgent not inheriting BaseAgent | `structured_agent.py` | Pending — different API (`invoke` vs `run`) |
| No state reducers (LangGraph-style) | `base.py` | Pending — flat AgentState |
| Single LLM provider | `mistral_provider.py` | Pending — OpenAI/Anthropic stubs |

### Implemented Improvements

| Improvement | Module | Description |
|-------------|--------|-------------|
| **AgentProtocol** | `core/protocol.py` | Unified protocol for all agents |
| **RouterAgent** | `router/router_agent.py` | LLM-based agent routing; `route_only()` for Ensemble |
| **PlannerAgent** | `planner/planner_agent.py` | Task decomposition → executor |
| **WorkflowGraph** | `graph/workflow.py` | Composable graph workflows with edges |
| **TraceHooks** | `tracing/hooks.py` | Observability events (run, LLM, tool) |
| **Ensemble router** | `orchestration/ensemble.py` | Optional `router` param for LLM-based agent selection |
| **RunConfig retry_on** | `runner.py` | Configurable retry exception types |

---

## 2. Recommended Architecture

```
agentensemble/
├── agents/           # ReAct, StateGraph, RAG, Hybrid, Structured
├── core/             # AgentProtocol, RunnableProtocol
├── llm/              # LLMProvider, MistralLLMProvider
├── memory/           # Session, InMemorySession, SQLiteSession
├── orchestration/    # Ensemble, Supervisor, Swarm, Pipeline, Debate
├── tools/            # Tool protocol, @function_tool, built-ins
├── router/           # RouterAgent
├── planner/          # PlannerAgent
├── graph/            # WorkflowGraph
├── tracing/          # TraceHooks, TraceEvent
├── testing/          # Benchmark, AgentComparison, Metrics
└── runner.py         # Runner, RunConfig, RunHooks
```

---

## 3. Missing Features to Implement

### High Priority

1. **StructuredAgent alignment** — Inherit BaseAgent or implement AgentProtocol; unify `run()`/`arun()` and return shape
2. **Async Session** — `Session` protocol with `aget_messages`, `aadd_messages` for async agents
3. **Human-in-the-loop** — `interrupt_before_tools` in RunConfig; pause before sensitive tools, resume with approval
4. **Cost-aware routing** — Model router that selects cheaper models for simple tasks

### Medium Priority

5. **OpenAI/Anthropic providers** — Implement `LLMProvider` for multi-provider support
6. **State reducers** — LangGraph-style `Annotated[list, append_reducer]` for parallel node merges
7. **Tool approval flow** — `@function_tool(needs_approval=True)` for payment, file writes
8. **Session compaction** — Summarize old messages to fit context window (Memoria-style)

### Lower Priority

9. **MCP integration** — `MCPToolAdapter` for dynamic tool discovery
10. **OpenTelemetry** — Export TraceHooks to OTel spans
11. **Benchmark suite** — Automated regression tests with real metrics

---

## 4. Suggested Improvements to Existing Modules

### ReActAgent

- Allow multi-round tool use after tool results (currently omits tools on second call)
- Expose token usage from `LLMResponse.usage` in metadata

### HybridAgent

- Add `arun()` support for async orchestration (already done)
- Consider LLM-based synthesis for `_synthesize_answer`

### Ensemble

- Use `router` for supervisor mode (implemented)
- Add `DebateOrchestrator` integration with real aggregator voting

### Tools

- Add native `arun()` to SearchTool, RAGTool for true async
- ToolRegistry integration with agents (agents currently use raw `tools` list)

### Memory

- SQLiteSession: configurable DB path, multi-tenancy
- Add `RedisSession` for distributed deployments

---

## 5. New Modules Added

| Module | Purpose |
|--------|---------|
| `agentensemble.core` | AgentProtocol, RunnableProtocol |
| `agentensemble.router` | RouterAgent for LLM-based routing |
| `agentensemble.planner` | PlannerAgent for task decomposition |
| `agentensemble.graph` | WorkflowGraph for composable workflows |
| `agentensemble.tracing` | TraceHooks, TraceEvent for observability |

---

## 6. Production Readiness Checklist

- [x] Runner with RunConfig, hooks, retries
- [x] Session protocol (InMemory, SQLite)
- [x] Async API on all agents
- [x] Benchmark and AgentComparison
- [ ] Structured error types (RunError, ToolError, LLMError)
- [ ] Rate limit retry with backoff
- [ ] Input/output guardrails
- [ ] OpenTelemetry export
- [ ] Security: no secrets in logs, sanitized tool args

---

## 7. Example Additions

| Example | Path | Purpose |
|---------|------|---------|
| Research Agent | `examples/research_agent_example.py` | Planner, Router, Ensemble |
| Graph Workflow | `examples/graph_workflow_example.py` | WorkflowGraph |
| Runner | `examples/runner_example.py` | RunConfig, RunHooks |
