from src import tools
from src.agent import TOOLS_SCHEMA


def test_every_schema_entry_has_the_required_keys():
    for entry in TOOLS_SCHEMA:
        assert "name" in entry
        assert "description" in entry
        assert "input_schema" in entry

        input_schema = entry["input_schema"]
        assert input_schema["type"] == "object"
        assert "properties" in input_schema
        assert "required" in input_schema


def test_every_schema_name_matches_a_real_function_in_tools_py():
    for entry in TOOLS_SCHEMA:
        assert hasattr(tools, entry["name"]), (
            f"TOOLS_SCHEMA references '{entry['name']}', "
            f"but tools.py has no such function."
        )


def test_load_dataset_is_deliberately_not_exposed_as_a_tool():
    schema_names = {entry["name"] for entry in TOOLS_SCHEMA}
    assert "load_dataset" not in schema_names
