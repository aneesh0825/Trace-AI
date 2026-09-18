"""
Core agent loop for Trace.

An LLM-driven tool-calling loop: instead of main.py calling tools.py
functions in a fixed order, Claude decides which tool to call and when,
based on the tool schemas below.
"""

import json

from src import tools
from src.state import AgentState


TOOLS_SCHEMA = [
    {
        "name": "summarize_dataset",
        "description": (
            "Get basic structural information about the dataset: row "
            "count, column count, column names, data types, and "
            "missing-value counts per column. Use this first to "
            "understand what's in the data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "numeric_summary",
        "description": (
            "Get summary statistics (count, mean, std, min, quartiles, "
            "max) for every numeric column in the dataset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
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
            "required": ["column"],
        },
    },
    {
        "name": "correlation_matrix",
        "description": (
            "Get pairwise correlations between all numeric columns in "
            "the dataset. Use this to spot relationships between "
            "numeric variables."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "compare_segment",
        "description": (
            "Compare a segment of the data against the rest of the "
            "dataset, for every numeric column. Use this to check "
            "whether a specific value in a column (e.g. a particular "
            "month or region) behaves differently from the rest of the "
            "data - returns each numeric column's segment_mean, "
            "baseline_mean, difference, and percent_change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
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
            "required": ["filter_column", "filter_value"],
        },
    },
]


TOOL_FUNCTIONS = {
    "summarize_dataset": tools.summarize_dataset,
    "numeric_summary": tools.numeric_summary,
    "value_counts": tools.value_counts,
    "correlation_matrix": tools.correlation_matrix,
    "compare_segment": tools.compare_segment,
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


def execute_tool(name, tool_input, df):
    """
    Run a tool by name against the loaded DataFrame, using the arguments
    Claude provided. Never raises: any failure (unknown tool, bad
    arguments, an exception inside the tool) becomes an is_error result
    instead, so the agent loop can hand it back to Claude and keep going.

    Returns {"content": str, "is_error": bool} - the shape the agent loop
    needs to build an Anthropic tool_result block.
    """
    function = TOOL_FUNCTIONS.get(name)

    if function is None:
        return {
            "content": f"Unknown tool: '{name}'",
            "is_error": True,
        }

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


SYSTEM_PROMPT = (
    "You are Trace, an autonomous data investigation agent. You have "
    "tools for exploring a tabular dataset. Use them as needed to answer "
    "the user's question, then give a clear, concise final answer "
    "summarizing what you found. Don't guess at data you haven't queried."
)


def run_agent_loop(client, state, model="claude-sonnet-5", max_turns=8):
    """
    Drive the tool-calling loop: ask Claude what to do next, run any
    tools it requests against state.df, feed the results back, and
    repeat until Claude answers with plain text (stop_reason != "tool_use")
    or max_turns is reached. Mutates and returns state.
    """
    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
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
            result = execute_tool(block.name, block.input, state.df)
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


if __name__ == "__main__":
    import anthropic
    from dotenv import load_dotenv

    from src.tools import load_dataset

    load_dotenv()  # picks up ANTHROPIC_API_KEY from a local .env, if present

    DATA_PATH = "data/sample_sales.csv"
    QUESTION = (
        "Investigate this sales dataset. I noticed revenue dropped in "
        "August - figure out what's going on and explain it to me."
    )

    client = anthropic.Anthropic()
    df = load_dataset(DATA_PATH)
    state = AgentState(df=df, messages=[{"role": "user", "content": QUESTION}])

    final_state = run_agent_loop(client, state)

    final_message = final_state.messages[-1]
    final_text = "\n".join(
        block.text for block in final_message["content"] if block.type == "text"
    )
    print(final_text)
