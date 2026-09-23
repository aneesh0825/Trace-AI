import json

import pandas as pd
import pytest

from src.agent import get_latest_answer_text
from src.graph import record_hypothesis, update_hypothesis
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


# --- investigation graph round-trip ---


def test_save_writes_graph_into_the_manifest(tmp_path, datasets, full_history):
    graph = []
    record_hypothesis(graph, "August drop is a west region issue.", turn=1)
    state = AgentState(datasets=datasets, messages=full_history, graph=graph)
    path = tmp_path / "investigation"

    save_investigation(state, path)

    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["graph"] == graph


def test_round_trip_preserves_graph_contents(tmp_path, datasets, full_history):
    graph = []
    record_hypothesis(graph, "August drop is a west region issue.", turn=1)
    update_hypothesis(
        graph,
        "h1",
        status="supported",
        evidence={
            "tool_name": "compare_segment",
            "note": "West region revenue down 40% vs baseline.",
        },
        turn=2,
    )
    state = AgentState(datasets=datasets, messages=full_history, graph=graph)
    path = tmp_path / "investigation"

    save_investigation(state, path)
    loaded = load_investigation(path)

    assert loaded.graph == graph


def test_round_trip_with_empty_graph_stays_empty(tmp_path, datasets, full_history):
    state = AgentState(datasets=datasets, messages=full_history)
    path = tmp_path / "investigation"

    save_investigation(state, path)
    loaded = load_investigation(path)

    assert loaded.graph == []


def test_load_investigation_handles_a_manifest_saved_before_the_graph_existed(
    tmp_path, datasets, full_history
):
    # Simulates a pre-V4 saved investigation: manifest.json has no "graph"
    # key at all. Loading it shouldn't crash - it should just come back
    # with an empty graph.
    state = AgentState(datasets=datasets, messages=full_history)
    path = tmp_path / "investigation"
    save_investigation(state, path)

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["graph"]
    manifest_path.write_text(json.dumps(manifest))

    loaded = load_investigation(path)

    assert loaded.graph == []
