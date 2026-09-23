import pytest

from src.graph import record_hypothesis, update_hypothesis


def test_record_hypothesis_creates_entry_with_proposed_status():
    graph = []

    entry = record_hypothesis(graph, "August drop is a west region issue.", turn=1)

    assert entry == {
        "id": "h1",
        "statement": "August drop is a west region issue.",
        "status": "proposed",
        "evidence": [],
        "related_to": None,
        "created_turn": 1,
        "updated_turn": 1,
    }
    assert graph == [entry]


def test_record_hypothesis_generates_sequential_ids():
    graph = []

    record_hypothesis(graph, "First hypothesis.")
    second = record_hypothesis(graph, "Second hypothesis.")

    assert [h["id"] for h in graph] == ["h1", "h2"]
    assert second["id"] == "h2"


def test_record_hypothesis_rejects_empty_statement():
    graph = []

    with pytest.raises(ValueError):
        record_hypothesis(graph, "   ")


def test_record_hypothesis_rejects_unknown_related_to():
    graph = []

    with pytest.raises(ValueError):
        record_hypothesis(graph, "New hypothesis.", related_to="h99")


def test_record_hypothesis_accepts_valid_related_to():
    graph = []
    first = record_hypothesis(graph, "Seasonal dip.")

    second = record_hypothesis(
        graph, "Actually a west region issue.", related_to=first["id"]
    )

    assert second["related_to"] == "h1"


def test_update_hypothesis_changes_status():
    graph = []
    record_hypothesis(graph, "August drop is seasonal.", turn=1)

    updated = update_hypothesis(graph, "h1", status="refuted", turn=2)

    assert updated["status"] == "refuted"
    assert graph[0]["status"] == "refuted"


def test_update_hypothesis_rejects_invalid_status():
    graph = []
    record_hypothesis(graph, "August drop is seasonal.")

    with pytest.raises(ValueError):
        update_hypothesis(graph, "h1", status="very_sure")


def test_update_hypothesis_rejects_unknown_id():
    graph = []

    with pytest.raises(ValueError):
        update_hypothesis(graph, "h1", status="supported")


def test_update_hypothesis_appends_evidence():
    graph = []
    record_hypothesis(graph, "August drop is a west region issue.")
    evidence = {
        "tool_use_id": "t1",
        "tool_name": "compare_segment",
        "note": "West region revenue down 40% vs baseline in August.",
    }

    updated = update_hypothesis(graph, "h1", evidence=evidence)

    assert updated["evidence"] == [evidence]


def test_update_hypothesis_can_update_status_and_evidence_together():
    graph = []
    record_hypothesis(graph, "August drop is a west region issue.")
    evidence = {
        "tool_use_id": "t1",
        "tool_name": "compare_segment",
        "note": "West region revenue down 40% vs baseline in August.",
    }

    updated = update_hypothesis(graph, "h1", status="supported", evidence=evidence)

    assert updated["status"] == "supported"
    assert updated["evidence"] == [evidence]


def test_update_hypothesis_accumulates_multiple_evidence_entries():
    graph = []
    record_hypothesis(graph, "August drop is a west region issue.")
    first_evidence = {"tool_use_id": "t1", "tool_name": "compare_segment", "note": "First."}
    second_evidence = {"tool_use_id": "t2", "tool_name": "run_sql", "note": "Second."}

    update_hypothesis(graph, "h1", evidence=first_evidence)
    updated = update_hypothesis(graph, "h1", evidence=second_evidence)

    assert updated["evidence"] == [first_evidence, second_evidence]


def test_update_hypothesis_updates_updated_turn_but_not_created_turn():
    graph = []
    record_hypothesis(graph, "August drop is a west region issue.", turn=1)

    updated = update_hypothesis(graph, "h1", status="supported", turn=5)

    assert updated["created_turn"] == 1
    assert updated["updated_turn"] == 5


def test_update_hypothesis_with_no_status_or_evidence_only_touches_turn():
    graph = []
    record_hypothesis(graph, "August drop is a west region issue.", turn=1)

    updated = update_hypothesis(graph, "h1", turn=2)

    assert updated["status"] == "proposed"
    assert updated["evidence"] == []
    assert updated["updated_turn"] == 2
