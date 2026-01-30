from typing import Optional

from agno.agent import Agent
from agno.eval.reliability import ReliabilityEval, ReliabilityResult
from agno.models.ollama import Ollama
from agno.run.agent import RunOutput
from agno.tools.duckduckgo import DuckDuckGoTools

OLLAMA_HOST = "http://localhost:11434"


def web_agent_reliability():
    agent = Agent(
        name="Web Research Agent",
        model=Ollama(id="llama3.2", host=OLLAMA_HOST),
        tools=[DuckDuckGoTools()],
        instructions=[
            "You are a web research agent.",
            "For today's news or current events, you MUST use web search.",
            "Always fetch fresh information before responding.",
            "Do not answer from prior knowledge.",
        ],
    )

    response: RunOutput = agent.run(
        "Use web search to fetch and summarize today's AI news."
    )

    evaluation = ReliabilityEval(
        name="Web Agent Tool Reliability",
        agent_response=response,
        expected_tool_calls=[
            "web_search",
        ],
    )

    result: Optional[ReliabilityResult] = evaluation.run(print_results=True)
    result.assert_passed()


if __name__ == "__main__":
    web_agent_reliability()
