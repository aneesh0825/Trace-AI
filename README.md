# Trace

**Give it a problem. It investigates.**

Trace is an autonomous data-investigation agent. Instead of answering a question directly, it behaves like an analyst: it forms competing hypotheses, runs real statistical tests to check them, weighs the evidence, and gives a conclusion it can defend — or revise when challenged.

## What it does

Give Trace a dataset and a problem, e.g. *"Revenue dropped in August — figure out why."* Trace will:

1. **Form hypotheses** about what could explain the problem
2. **Run analysis** — deciding on its own which tool to use (summary statistics, correlations, segment comparisons, or SQL queries) rather than following a fixed script
3. **Weigh the evidence** and give a clear, defensible conclusion
4. **Accept a challenge** — push back on its conclusion and it runs a new analysis to check itself, instead of just arguing in text
5. **Save and resume** — pause an investigation and pick it back up later with full context intact

This was verified end-to-end on real sales data: Trace correctly identified that an August revenue drop was concentrated in one region/channel combination (not a company-wide slowdown), and when challenged on its initial explanation, re-investigated by region and confirmed a more precise root cause.

## Architecture

Trace is built around a tool-calling loop, not a chatbot wrapper:

```
User question
      │
      ▼
Claude decides which tool to call (or answers directly)
      │
      ▼
execute_tool() dispatches to the real Python function,
catching bad arguments/errors instead of crashing
      │
      ▼
Result fed back to Claude as a tool_result
      │
      ▼
Repeat until Claude gives a final answer
```

- **`src/agent.py`** — the core loop (`run_agent_loop`), the tool dispatcher (`execute_tool`), and the tool schema Claude reads to decide what's available
- **`src/tools.py`** — the actual analysis functions: dataset summaries, correlations, segment comparisons, and SQL execution (via an in-memory SQLite table)
- **`src/state.py`** — `AgentState`: the datasets loaded and the running conversation history
- **`src/persistence.py`** — save/load an investigation to disk (datasets as CSV, conversation as JSON)
- **`continue_investigation()`** — lets a user challenge a conclusion; reuses the full conversation history so Trace can revise its answer with context, not from scratch

Every piece of branching logic (unknown tool names, bad arguments, empty query results, resuming a saved conversation) is covered by tests — 54 passing at last count, built test-first throughout.

## Tech stack

- **Python** — pandas for analysis, stdlib `sqlite3` for SQL execution
- **Anthropic API** (Claude Sonnet 5) — tool-calling / `tool_use` for the agent loop
- **pytest** — all core logic test-driven

## Running it

```bash
pip install -r requirements.txt
# create a .env file with ANTHROPIC_API_KEY=sk-ant-...
python -m src.agent                       # start a new investigation
python -m src.agent path/to/saved/folder  # resume a saved one
```

Inside the follow-up loop, type `save <path>` to save the investigation, or `done` to exit.

## Roadmap

**Built:**
- Tool-calling agent core
- Challenge & re-investigate loop
- Multi-dataset support & SQL execution
- Investigation save & resume

**Planned:**
- Investigation graph — hypotheses and evidence as a visual tree
- Skeptic agent — a second AI that pressure-tests every finding
- Autopilot — surfaces findings without a question being asked
- Full web product — accounts, saved investigations, a live UI
