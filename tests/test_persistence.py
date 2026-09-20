import json

import pandas as pd
import pytest

from src.agent import get_latest_answer_text
from src.persistence import load_investigation, save_investigation
from src.state import AgentState
from tests.fakes import text_block, tool_use_block


@pytest.fixture
def datasets():
    return {
        "sales": pd.DataFrame(
            {
                "revenue": [100, 200, 300],
                "region": ["west", "east", "west"],
            }
        ),
        "returns": pd.DataFrame({"amount": [10, 20]}),
    }


@pytest.fixture
def full_history():
    return [
        {"role": "user", "content": "Investigate revenue."},
        {
            "role": "assistant",
            "content": [
                tool_use_block("t1", "summarize_dataset", {"dataset_name": "sales"})
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": json.dumps({"rows": 3}),
                    "is_error": False,
                }
            ],
        },
        {"role": "assistant", "content": [text_block("There are 3 rows.")]},
    ]


def test_save_creates_manifest_and_dataset_csvs(tmp_path, datasets, full_history):
    state = AgentState(datasets=datasets, messages=full_history)
    path = tmp_path / "investigation"

    save_investigation(state, path)

    assert (path / "manifest.json").exists()
    assert (path / "datasets" / "sales.csv").exists()
    assert (path / "datasets" / "returns.csv").exists()


def test_round_trip_preserves_dataset_contents(tmp_path, datasets, full_history):
    state = AgentState(datasets=datasets, messages=full_history)
    path = tmp_path / "investigation"

    save_investigation(state, path)
    loaded = load_investigation(path)

    pd.testing.assert_frame_equal(loaded.datasets["sales"], datasets["sales"])
    pd.testing.assert_frame_equal(loaded.datasets["returns"], datasets["returns"])


def test_multiple_datasets_round_trip_independently(tmp_path, datasets, full_history):
    state = AgentState(datasets=datasets, messages=full_history)
    path = tmp_path / "investigation"

    save_investigation(state, path)
    loaded = load_investigation(path)

    assert set(loaded.datasets.keys()) == {"sales", "returns"}


def test_round_trip_preserves_plain_user_messages(tmp_path, datasets, full_history):
    state = AgentState(datasets=datasets, messages=full_history)
    path = tmp_path / "investigation"

    save_investigation(state, path)
    loaded = load_investigation(path)

    assert loaded.messages[0] == {"role": "user", "content": "Investigate revenue."}


def test_round_trip_converts_assistant_text_block_to_dict(
    tmp_path, datasets, full_history
):
    state = AgentState(datasets=datasets, messages=full_history)
    path = tmp_path / "investigation"

    save_investigation(state, path)
    loaded = load_investigation(path)

    final_block = loaded.messages[-1]["content"][0]
    assert final_block == {"type": "text", "text": "There are 3 rows."}


def test_round_trip_converts_tool_use_block_to_dict(tmp_path, datasets, full_history):
    state = AgentState(datasets=datasets, messages=full_history)
    path = tmp_path / "investigation"

    save_investigation(state, path)
    loaded = load_investigation(path)

    tool_use = loaded.messages[1]["content"][0]
    assert tool_use == {
        "type": "tool_use",
        "id": "t1",
        "name": "summarize_dataset",
        "input": {"dataset_name": "sales"},
    }


def test_round_trip_preserves_tool_result_messages_unchanged(
    tmp_path, datasets, full_history
):
    state = AgentState(datasets=datasets, messages=full_history)
    path = tmp_path / "investigation"

    save_investigation(state, path)
    loaded = load_investigation(path)

    tool_result = loaded.messages[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "t1"
    assert tool_result["is_error"] is False
    assert json.loads(tool_result["content"])["rows"] == 3


def test_get_latest_answer_text_works_immediately_after_load(
    tmp_path, datasets, full_history
):
    state = AgentState(datasets=datasets, messages=full_history)
    path = tmp_path / "investigation"

    save_investigation(state, path)
    loaded = load_investigation(path)

    assert get_latest_answer_text(loaded) == "There are 3 rows."


def test_save_accepts_a_plain_string_path(tmp_path, datasets, full_history):
    state = AgentState(datasets=datasets, messages=full_history)
    path = str(tmp_path / "investigation")

    save_investigation(state, path)
    loaded = load_investigation(path)

    assert loaded.datasets["sales"].shape == datasets["sales"].shape
