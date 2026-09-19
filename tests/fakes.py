"""
Shared test doubles for exercising the agent loop without hitting the
real Anthropic API. Mimics just enough of the SDK's response shape
(stop_reason + content blocks with .type) to drive run_agent_loop.
"""

from types import SimpleNamespace


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(tool_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def response(content_blocks, stop_reason):
    return SimpleNamespace(stop_reason=stop_reason, content=content_blocks)


class FakeMessages:
    """Stands in for client.messages - returns pre-scripted responses in order."""

    def __init__(self, scripted_responses):
        self._scripted_responses = list(scripted_responses)
        self.calls = []

    def create(self, **kwargs):
        # Snapshot messages at call time - the caller keeps appending to
        # the same list afterward, so storing the reference itself would
        # make this record retroactively "change" as the conversation
        # continues.
        if "messages" in kwargs:
            kwargs = {**kwargs, "messages": list(kwargs["messages"])}
        self.calls.append(kwargs)
        return self._scripted_responses.pop(0)


class FakeClient:
    def __init__(self, scripted_responses):
        self.messages = FakeMessages(scripted_responses)
