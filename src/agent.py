"""
Core agent loop for Trace.

An LLM-driven tool-calling loop: instead of main.py calling tools.py
functions in a fixed order, Claude decides which tool to call and when,
based on the tool schemas below.
"""

import json

from src import graph as graph_module
from src import tools
from src.state import AgentState


DATASET_NAME_PROPERTY = {
    "type": "string",
    "description": (
        "The name of the dataset to run this tool against (see the "
        "list of available datasets in the system prompt)."
    ),
}


TOOLS_SCHEMA = [
    {
        "name": "summarize_dataset",
        "description": (
            "Get basic structural information about a dataset: row "
            "count, column count, column names, data types, and "
            "missing-value counts per column. Use this first to "
            "understand what's in the data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": ["dataset_name"],
        },
    },
    {
        "name": "numeric_summary",
        "description": (
            "Get summary statistics (count, mean, std, min, quartiles, "
            "max) for every numeric column in a dataset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": ["dataset_name"],
        },
    },
    {
        "name": "value_counts",
        "description": (
            "Get the most common values in a single column, with their "
            "counts. Use this to understand the distribution of a "
            "categorical column."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": DATASET_NAME_PROPERTY,
                "column": {
                    "type": "string",
                    "description": "The name of the column to count values in.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "How many of the most common values to return.",
                    "default": 10,
                },
            },
            "required": ["dataset_name", "column"],
        },
    },
    {
        "name": "correlation_matrix",
        "description": (
            "Get pairwise correlations between all numeric columns in "
            "a dataset. Use this to spot relationships between numeric "
            "variables."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": ["dataset_name"],
        },
    },
    {
        "name": "compare_segment",
        "description": (
            "Compare a segment of a dataset against the rest of that "
            "dataset, for every numeric column. Use this to check "
            "whether a specific value in a column (e.g. a particular "
            "month or region) behaves differently from the rest of the "
            "data - returns each numeric column's segment_mean, "
            "baseline_mean, difference, and percent_change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": DATASET_NAME_PROPERTY,
                "filter_column": {
                    "type": "string",
                    "description": (
                        "The column to segment by (e.g. 'month', 'region')."
                    ),
                },
                "filter_value": {
                    "type": ["string", "number"],
                    "description": (
                        "The value that defines the segment (e.g. "
                        "'August', 'west'). Rows where filter_column "
                        "equals this value are compared against all "
                        "other rows."
                    ),
                },
            },
            "required": ["dataset_name", "filter_column", "filter_value"],
        },
    },
    {
        "name": "run_sql",
        "description": (
            "Run a read-only SQL SELECT query against a dataset, for "
            "filtering, aggregation, or grouping that the other tools "
            "don't cover directly. The dataset is available in the "
            "query as a table named 'dataset' (e.g. \"SELECT region, "
            "SUM(revenue) AS total FROM dataset GROUP BY region\"). "
            "Only a single SELECT (or WITH ... SELECT) statement is "
            "allowed, and results are capped at 200 rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": DATASET_NAME_PROPERTY,
                "query": {
                    "type": "string",
                    "description": (
                        "The SQL SELECT query to run. Reference the "
                        "dataset as the table 'dataset'."
                    ),
                },
            },
            "required": ["dataset_name", "query"],
        },
    },
    {
        "name": "record_hypothesis",
        "description": (
            "Record a new hypothesis about what's driving a pattern in "
            "the data, before you test it. Use this to make your "
            "reasoning explicit and trackable, separate from your "
            "analysis calls. Only use this for a genuinely new "
            "hypothesis - if new evidence bears on one you already "
            "recorded, use update_hypothesis instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "statement": {
                    "type": "string",
                    "description": (
                        "The hypothesis, stated as a specific, testable "
                        "claim (e.g. 'The August revenue drop is driven "
                        "by the west region')."
                    ),
                },
                "related_to": {
                    "type": "string",
                    "description": (
                        "The id of an existing hypothesis this one "
                        "refines, supersedes, or contradicts, if any "
                        "(e.g. 'h1')."
                    ),
                },
            },
            "required": ["statement"],
        },
    },
    {
        "name": "update_hypothesis",
        "description": (
            "Update a hypothesis you already recorded, after gathering "
            "evidence for or against it: change its status and/or "
            "attach what you found. Use this instead of "
            "record_hypothesis when new evidence bears on an existing "
            "hypothesis rather than suggesting a new one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis_id": {
                    "type": "string",
                    "description": (
                        "The id of the hypothesis to update (e.g. 'h1')."
                    ),
                },
                "status": {
                    "type": "string",
                    "enum": sorted(graph_module.STATUSES),
                    "description": "The hypothesis's new status, if it changed.",
                },
                "evidence": {
                    "type": "object",
                    "description": (
                        "What you found, if you ran a test for this "
                        "hypothesis."
                    ),
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": (
                                "The analysis tool that produced this "
                                "evidence (e.g. 'compare_segment')."
                            ),
                        },
                        "tool_use_id": {
                            "type": "string",
                            "description": (
                                "The id of that tool call, if you have "
                                "it, so the evidence can be traced back "
                                "to its raw result."
                            ),
                        },
                        "note": {
                            "type": "string",
                            "description": (
                                "A short, specific summary of what the "
                                "evidence showed."
                            ),
                        },
                    },
                    "required": ["note"],
                },
            },
            "required": ["hypothesis_id"],
        },
    },
]


DATASET_TOOL_FUNCTIONS = {
    "summarize_dataset": tools.summarize_dataset,
    "numeric_summary": tools.numeric_summary,
    "value_counts": tools.value_counts,
    "correlation_matrix": tools.correlation_matrix,
    "compare_segment": tools.compare_segment,
    "run_sql": tools.run_sql,
}

GRAPH_TOOL_FUNCTIONS = {
    "record_hypothesis": graph_module.record_hypothesis,
    "update_hypothesis": graph_module.update_hypothesis,
}


def _to_jsonable(value):
    """
    json.dumps default= hook: tools.py returns pandas/numpy scalars
    (numpy.int64, numpy.float64, ...) that the stdlib json module can't
    serialize on its own. Anything with .item() is such a scalar.
    """
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _execute_dataset_tool(name, function, tool_input, datasets):
    # Copy before popping: tool_input may be the same dict object stored
    # in state.messages (the tool_use block Claude sent) - mutating it in
    # place would silently corrupt that history.
    tool_input = dict(tool_input)
    dataset_name = tool_input.pop("dataset_name", None)

    if dataset_name not in datasets:
        available = ", ".join(sorted(datasets)) or "(none loaded)"
        return {
            "content": (
                f"Unknown dataset: {dataset_name!r}. "
                f"Available datasets: {available}."
            ),
            "is_error": True,
        }

    df = datasets[dataset_name]

    try:
        result = function(df, **tool_input)
    except Exception as error:
        return {
            "content": f"Error running '{name}': {error}",
            "is_error": True,
        }

    return {
        "content": json.dumps(result, default=_to_jsonable),
        "is_error": False,
    }


def _execute_graph_tool(name, function, tool_input, graph, turn):
    # turn is harness bookkeeping, not something Claude supplies - stamp
    # it ourselves rather than trusting it in tool_input.
    tool_input = dict(tool_input)
    tool_input.pop("turn", None)

    try:
        result = function(graph, turn=turn, **tool_input)
    except Exception as error:
        return {
            "content": f"Error running '{name}': {error}",
            "is_error": True,
        }

    return {
        "content": json.dumps(result, default=_to_jsonable),
        "is_error": False,
    }


def execute_tool(name, tool_input, state, turn=None):
    """
    Run a tool by name against agent state, using the arguments Claude
    provided. Never raises: any failure (unknown tool, unknown/missing
    dataset_name, bad arguments, an exception inside the tool) becomes
    an is_error result instead, so the agent loop can hand it back to
    Claude and keep going.

    Dataset tools (summarize_dataset, run_sql, ...) read state.datasets
    and require dataset_name. Graph tools (record_hypothesis,
    update_hypothesis) mutate state.graph instead - turn is stamped by
    the caller (the agent loop), not supplied by Claude.

    Returns {"content": str, "is_error": bool} - the shape the agent loop
    needs to build an Anthropic tool_result block.
    """
    if name in GRAPH_TOOL_FUNCTIONS:
        return _execute_graph_tool(
            name, GRAPH_TOOL_FUNCTIONS[name], tool_input, state.graph, turn
        )

    function = DATASET_TOOL_FUNCTIONS.get(name)

    if function is None:
        return {
            "content": f"Unknown tool: '{name}'",
            "is_error": True,
        }

    return _execute_dataset_tool(name, function, tool_input, state.datasets)


def build_system_prompt(datasets):
    dataset_names = ", ".join(sorted(datasets)) or "(none loaded)"
    return (
        "You are Trace, an autonomous data investigation agent. You have "
        "tools for exploring tabular datasets. Available datasets: "
        f"{dataset_names}. Every tool call must specify which dataset it "
        "applies to via dataset_name. Use the tools as needed to answer "
        "the user's question, then give a clear, concise final answer "
        "summarizing what you found. Don't guess at data you haven't "
        "queried.\n\n"
        "As you investigate, keep the investigation graph in sync with "
        "what you actually find - this is a required step, not "
        "optional documentation. For every hypothesis: call "
        "record_hypothesis before you test it; then, as soon as a tool "
        "result gives you evidence for or against it, call "
        "update_hypothesis in that same turn - don't just describe the "
        "finding in your answer and move on. Update an existing "
        "hypothesis when new evidence bears on it; only record a new "
        "one for a genuinely different explanation. Before giving your "
        "final answer, check every hypothesis you've recorded: if any "
        "is still 'proposed' despite evidence you've already gathered, "
        "call update_hypothesis on it first."
    )


def run_agent_loop(client, state, model="claude-sonnet-5", max_turns=8):
    """
    Drive the tool-calling loop: ask Claude what to do next, run any
    tools it requests against state (datasets for analysis tools, graph
    for hypothesis tools), feed the results back, and repeat until
    Claude answers with plain text (stop_reason != "tool_use") or
    max_turns is reached. Mutates and returns state.
    """
    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=build_system_prompt(state.datasets),
            tools=TOOLS_SCHEMA,
            messages=state.messages,
        )
        state.messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return state

        tool_use_blocks = [
            block for block in response.content if block.type == "tool_use"
        ]

        tool_results = []
        for block in tool_use_blocks:
            result = execute_tool(block.name, block.input, state, turn=turn)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result["content"],
                    "is_error": result["is_error"],
                }
            )

        state.messages.append({"role": "user", "content": tool_results})

    return state


def continue_investigation(client, state, follow_up_text, **kwargs):
    """
    Continue the same conversation with a follow-up message (e.g. "I
    don't buy this, check region instead"), reusing state.messages so
    Claude has full context of what it already found. Delegates to
    run_agent_loop, so a follow-up can trigger new tool calls exactly
    like the initial investigation did. kwargs (model, max_turns) pass
    straight through to run_agent_loop.
    """
    state.messages.append({"role": "user", "content": follow_up_text})
    return run_agent_loop(client, state, **kwargs)


DONE_WORDS = {"", "done", "exit", "quit"}


def _block_type(block):
    """
    A content block is an SDK object (attribute access) when it was just
    appended by run_agent_loop this session, or a plain dict (key
    access) when it came back from load_investigation. Handle both.
    """
    return block["type"] if isinstance(block, dict) else block.type


def _block_text(block):
    return block["text"] if isinstance(block, dict) else block.text


def get_latest_answer_text(state):
    """
    Extract the text from Claude's most recent message. If the loop
    stopped mid-investigation (max_turns reached while still calling
    tools), the last message is a tool_result, not text - flag that
    instead of crashing.
    """
    final_message = state.messages[-1]

    if final_message["role"] != "assistant":
        return "(Trace hit max_turns without giving a final answer.)"

    return "\n".join(
        _block_text(block)
        for block in final_message["content"]
        if _block_type(block) == "text"
    )


if __name__ == "__main__":
    import sys

    import anthropic
    from dotenv import load_dotenv

    from src.persistence import load_investigation, save_investigation
    from src.tools import load_dataset

    load_dotenv()  # picks up ANTHROPIC_API_KEY from a local .env, if present

    client = anthropic.Anthropic()

    resume_path = sys.argv[1] if len(sys.argv) > 1 else None

    if resume_path:
        state = load_investigation(resume_path)
        print(f"Resumed investigation from {resume_path}.")
        print(get_latest_answer_text(state))
    else:
        DATA_PATH = "data/sample_sales.csv"
        QUESTION = (
            "Investigate this sales dataset. I noticed revenue dropped in "
            "August - figure out what's going on and explain it to me."
        )

        datasets = {"sales": load_dataset(DATA_PATH)}
        state = AgentState(
            datasets=datasets, messages=[{"role": "user", "content": QUESTION}]
        )

        state = run_agent_loop(client, state)
        print(get_latest_answer_text(state))

    while True:
        follow_up = input(
            "\nFollow-up ('save <path>' to save, or 'done' to exit): "
        ).strip()

        if follow_up.lower() in DONE_WORDS:
            break

        if follow_up.lower().startswith("save"):
            parts = follow_up.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: save <path>")
                continue

            save_investigation(state, parts[1])
            print(f"Saved to {parts[1]}.")
            continue

        state = continue_investigation(client, state, follow_up)
        print(get_latest_answer_text(state))
