"""
Conversation state for the Trace agent loop.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AgentState:
    """
    datasets: every loaded dataset, keyed by the name Claude refers to it
        by in tool calls (e.g. {"sales": df}).
    messages: the running conversation history, in the shape the
        Anthropic Messages API expects (list of {"role", "content"} dicts).
    graph: the investigation graph - a structured record of hypotheses,
        evidence, and status, separate from the raw conversation in
        messages. See src/graph.py.
    """

    datasets: dict[str, pd.DataFrame]
    messages: list = field(default_factory=list)
    graph: list = field(default_factory=list)
