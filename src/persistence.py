"""
Save and load a Trace investigation (datasets + full message history) to
and from disk, so a conversation can be resumed later and continued with
continue_investigation as if it had never stopped.
"""

import json
from pathlib import Path

from src.state import AgentState
from src.tools import load_dataset


def _serialize_block(block):
    """
    Turn one message content block into a plain, JSON-safe dict.

    tool_result blocks built by execute_tool are already plain dicts and
    pass through unchanged. Assistant content blocks come straight from
    an Anthropic SDK response - pydantic models in production, plain
    SimpleNamespace objects in tests - and need converting.
    """
    if isinstance(block, dict):
        return block

    if hasattr(block, "model_dump"):
        return block.model_dump()

    return dict(vars(block))


def _serialize_messages(messages):
    serialized = []

    for message in messages:
        content = message["content"]

        if isinstance(content, list):
            content = [_serialize_block(block) for block in content]

        serialized.append({"role": message["role"], "content": content})

    return serialized


def save_investigation(state: AgentState, path) -> None:
    """
    Save state.datasets (one CSV per dataset) and state.messages (as
    JSON) under `path`, a directory created if it doesn't already exist.
    """
    directory = Path(path)
    datasets_dir = directory / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = {}
    for name, df in state.datasets.items():
        filename = f"{name}.csv"
        df.to_csv(datasets_dir / filename, index=False)
        dataset_files[name] = filename

    manifest = {
        "datasets": dataset_files,
        "messages": _serialize_messages(state.messages),
    }

    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_investigation(path) -> AgentState:
    """
    Rebuild an AgentState previously written by save_investigation.
    Datasets are reloaded through tools.load_dataset, the same function
    used the first time a dataset entered the system, so a saved-then-
    loaded dataset is identical to how it looked when it was saved.
    """
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text())

    datasets = {
        name: load_dataset(str(directory / "datasets" / filename))
        for name, filename in manifest["datasets"].items()
    }

    return AgentState(datasets=datasets, messages=manifest["messages"])
