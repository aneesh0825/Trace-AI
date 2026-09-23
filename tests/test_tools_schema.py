from src import graph, tools
from src.agent import GRAPH_TOOL_FUNCTIONS, TOOLS_SCHEMA

DATASET_SCHEMA_ENTRIES = [
    entry for entry in TOOLS_SCHEMA if entry["name"] not in GRAPH_TOOL_FUNCTIONS
]
GRAPH_SCHEMA_ENTRIES = [
    entry for entry in TOOLS_SCHEMA if entry["name"] in GRAPH_TOOL_FUNCTIONS
]


def test_every_schema_entry_has_the_required_keys():
    for entry in TOOLS_SCHEMA:
        assert "name" in entry
        assert "description" in entry
        assert "input_schema" in entry

        input_schema = entry["input_schema"]
        assert input_schema["type"] == "object"
        assert "properties" in input_schema
        assert "required" in input_schema


def test_every_dataset_schema_name_matches_a_real_function_in_tools_py():
    for entry in DATASET_SCHEMA_ENTRIES:
        assert hasattr(tools, entry["name"]), (
            f"TOOLS_SCHEMA references '{entry['name']}', "
            f"but tools.py has no such function."
        )


def test_every_graph_schema_name_matches_a_real_function_in_graph_py():
    for entry in GRAPH_SCHEMA_ENTRIES:
        assert hasattr(graph, entry["name"]), (
            f"TOOLS_SCHEMA references '{entry['name']}', "
            f"but graph.py has no such function."
        )


def test_load_dataset_is_deliberately_not_exposed_as_a_tool():
    schema_names = {entry["name"] for entry in TOOLS_SCHEMA}
    assert "load_dataset" not in schema_names


def test_every_dataset_tool_requires_a_dataset_name():
    for entry in DATASET_SCHEMA_ENTRIES:
        input_schema = entry["input_schema"]
        assert "dataset_name" in input_schema["properties"], (
            f"'{entry['name']}' has no dataset_name property."
        )
        assert "dataset_name" in input_schema["required"], (
            f"'{entry['name']}' doesn't require dataset_name."
        )


def test_no_graph_tool_requires_a_dataset_name():
    for entry in GRAPH_SCHEMA_ENTRIES:
        input_schema = entry["input_schema"]
        assert "dataset_name" not in input_schema["properties"], (
            f"'{entry['name']}' shouldn't take dataset_name - it operates "
            "on the investigation graph, not a dataset."
        )


def test_graph_tools_are_present_in_the_schema():
    schema_names = {entry["name"] for entry in TOOLS_SCHEMA}
    assert "record_hypothesis" in schema_names
    assert "update_hypothesis" in schema_names


def test_update_hypothesis_status_enum_matches_graph_statuses():
    entry = next(e for e in TOOLS_SCHEMA if e["name"] == "update_hypothesis")
    status_property = entry["input_schema"]["properties"]["status"]

    assert set(status_property["enum"]) == graph.STATUSES


def test_no_graph_tool_lets_claude_set_turn_directly():
    for entry in GRAPH_SCHEMA_ENTRIES:
        assert "turn" not in entry["input_schema"]["properties"], (
            f"'{entry['name']}' shouldn't expose turn - it's harness "
            "bookkeeping, stamped by the agent loop."
        )
