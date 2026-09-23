"""
The investigation graph: a structured record of the hypotheses Trace
forms, the evidence it gathers for each, and how its confidence in them
changes - separate from state.messages (the raw conversation).

A graph is a plain list of hypothesis dicts, mutated in place by
record_hypothesis / update_hypothesis. Hypotheses are never deleted,
only status-transitioned, so a refuted line of reasoning stays visible
instead of disappearing.
"""

STATUSES = {"proposed", "supported", "refuted", "strengthened", "weakened"}


def _find(graph, hypothesis_id):
    return next((h for h in graph if h["id"] == hypothesis_id), None)


def record_hypothesis(graph, statement, related_to=None, turn=None):
    """
    Add a new hypothesis to the graph with status "proposed". Mutates
    graph in place and returns the new entry.

    related_to, if given, must be the id of an existing hypothesis this
    one refines, supersedes, or contradicts - this is what makes the
    structure a graph rather than a flat list.
    """
    if not statement or not statement.strip():
        raise ValueError("statement is required.")

    if related_to is not None and _find(graph, related_to) is None:
        raise ValueError(f"Unknown related_to hypothesis id: {related_to!r}")

    entry = {
        "id": f"h{len(graph) + 1}",
        "statement": statement,
        "status": "proposed",
        "evidence": [],
        "related_to": related_to,
        "created_turn": turn,
        "updated_turn": turn,
    }
    graph.append(entry)
    return entry


def update_hypothesis(graph, hypothesis_id, status=None, evidence=None, turn=None):
    """
    Update an existing hypothesis: change its status, append an
    evidence entry, or both. Mutates the matching entry in place and
    returns it.
    """
    entry = _find(graph, hypothesis_id)
    if entry is None:
        raise ValueError(f"Unknown hypothesis id: {hypothesis_id!r}")

    if status is not None:
        if status not in STATUSES:
            raise ValueError(
                f"Invalid status: {status!r}. Must be one of {sorted(STATUSES)}."
            )
        entry["status"] = status

    if evidence is not None:
        entry["evidence"].append(evidence)

    entry["updated_turn"] = turn
    return entry
