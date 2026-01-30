from agno.agent import Agent
from agno.eval.performance import PerformanceEval
from agno.models.ollama import Ollama
from agno.tools.yfinance import YFinanceTools
from agno.tools.calculator import CalculatorTools

OLLAMA_HOST = "http://localhost:11434"


def run_finance_agent():
    agent = Agent(
        name="Finance Agent",
        model=Ollama(id="llama3.2", host=OLLAMA_HOST),
        tools=[
            YFinanceTools(),
            CalculatorTools(),
        ],
        instructions=[
            "When a stock, index, or company ticker is mentioned, evaluate if real market data is required.",
        "When the user asks for a CURRENT stock price, ALWAYS use YFinanceTools.",
        "Never answer stock prices from memory.",
        "If the ticker is unclear, ask for clarification.",
        "Explain finance concepts simply.",
        "Mention risks when discussing investments.",
        "Never use finance tools for self-questions.",
        ],
    )

    response = agent.run("What is the current stock price of AAPL?")
    print(f"Agent response: {response.content}")

    return response


finance_perf_eval = PerformanceEval(
    name="Finance Agent Performance Evaluation",
    func=run_finance_agent,
    num_iterations=3,   # run multiple times for avg latency
    warmup_runs=1,      # warmup to avoid cold-start bias
)


if __name__ == "__main__":
    finance_perf_eval.run(
        print_results=True,
        print_summary=True,
    )
