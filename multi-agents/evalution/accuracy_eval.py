from typing import Optional

from agno.agent import Agent
from agno.eval.accuracy import (
    AccuracyEval,
    AccuracyAgentResponse,
    AccuracyResult,
)
from agno.models.ollama import Ollama
from agno.tools.yfinance import YFinanceTools
from agno.tools.calculator import CalculatorTools

OLLAMA_HOST = "http://localhost:11434"


# -----------------------------
# Evaluator (Judge Agent)
# -----------------------------
evaluator_agent = Agent(
    name="Accuracy Evaluator",
    model=Ollama(id="llama3.2", host=OLLAMA_HOST),
    output_schema=AccuracyAgentResponse,
    instructions=[
        "You are an evaluation agent.",
        "Score accuracy from 0 to 10.",
        "Check factual correctness.",
        "Check correct tool usage.",
        "Be strict but fair.",
    ],
)


# -----------------------------
# Finance Agent (Under Test)
# -----------------------------
finance_agent = Agent(
    name="Finance Agent",
    model=Ollama(id="llama3.2", host=OLLAMA_HOST),
    tools=[
        YFinanceTools(),
        CalculatorTools(),
    ],
    instructions=[
        "When asked for CURRENT stock prices, always use YFinanceTools.",
        "Never answer stock prices from memory.",
        "Mention that stock prices can fluctuate.",
    ],
    markdown=True,
)


def run_finance_accuracy_eval():
    print("\n📊 Running Finance Agent Accuracy Evaluation\n")

    evaluation = AccuracyEval(
        model=Ollama(id="llama3.2", host=OLLAMA_HOST),
        agent=finance_agent,
        evaluator_agent=evaluator_agent,
        input="What is the current stock price of AAPL?",
        expected_output="A correct real-time stock price with explanation.",
        additional_guidelines=(
            "Answer must use live market data and mention price volatility."
        ),
    )

    result: Optional[AccuracyResult] = evaluation.run(print_results=True)

    assert result is not None, "No evaluation result returned"
    assert result.avg_score >= 7, "Finance Agent accuracy below threshold"


if __name__ == "__main__":
    run_finance_accuracy_eval()
    print("\n✅ Finance Agent accuracy evaluation passed")