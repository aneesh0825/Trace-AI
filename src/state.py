"""
Conversation state for the Trace agent loop.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AgentState:
    """
    df: the already-loaded dataset the tools operate on.
    messages: the running conversation history, in the shape the
        Anthropic Messages API expects (list of {"role", "content"} dicts).
    """

    df: pd.DataFrame
    messages: list = field(default_factory=list)
