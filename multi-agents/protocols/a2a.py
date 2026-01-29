from typing import TypedDict


class A2AMessage(TypedDict):
    from_agent: str
    to_agent: str
    task: str
    context: str
    expected_output: str
