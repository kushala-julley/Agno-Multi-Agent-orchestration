import os
from dotenv import load_dotenv
from pydantic import BaseModel

from agno.os import AgentOS
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.ollama import Ollama

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.calculator import CalculatorTools

# -------------------------------------------------
# Environment
# -------------------------------------------------
load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# -------------------------------------------------
# Shared memory
# -------------------------------------------------
db = SqliteDb(db_file="agno.db")

# -------------------------------------------------
# Common agent rules
# -------------------------------------------------
COMMON_INSTRUCTIONS = [
    "You are an AI agent running inside AgentOS.",
    "You have a fixed role and defined capabilities.",
    "Your special skills are research, analysis, reasoning, and summarization.",
    "If the user asks about your skills, tools, role, or how you work, answer directly about yourself.",
    "Do NOT explain artificial intelligence in general unless explicitly asked.",
    "Use tools ONLY when external real-world information is required.",
    "NEVER show tool calls, JSON, logs, or internal reasoning.",
    "Always give a clear, human-readable final answer.",
]

# -------------------------------------------------
# Web Agent 
# -------------------------------------------------
web_agent = Agent(
    name="Web Research Agent",
    role="Finds recent information and explains it clearly",
    model=Ollama(id="llama3.2", host=OLLAMA_HOST),
    tools=[
        DuckDuckGoTools(),
        WikipediaTools(),
    ],
   instructions=COMMON_INSTRUCTIONS + [
    # Tool selection rules (CRITICAL)
    "For recent news, current events, trends, or market impact, ALWAYS use DuckDuckGo.",
    "NEVER use Wikipedia for news, trends, or impact analysis.",
    "Use Wikipedia ONLY for static, encyclopedic facts (definitions, history, biographies).",
    "If a topic is not a clearly defined Wikipedia page, do NOT use Wikipedia.",

    # DuckDuckGo rules
    "When using DuckDuckGo, ALWAYS construct a clear, specific, non-empty search query.",
    "If a valid search query cannot be formed, do NOT call DuckDuckGo.",
    "Never guess or leave the search query empty.",
],

    db=db,
    add_history_to_context=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Analyzes stocks and financial trends",
    model=Ollama(id="llama3.2", host=OLLAMA_HOST),
    tools=[
        YFinanceTools(),
        CalculatorTools(),
    ],
    instructions=COMMON_INSTRUCTIONS + [
        "When a stock, index, or company ticker is mentioned, evaluate if real market data is required.",
        "When the user asks for a CURRENT stock price, ALWAYS use YFinanceTools.",
        "Never answer stock prices from memory.",
        "If the ticker is unclear, ask for clarification.",
        "Explain finance concepts simply.",
        "Mention risks when discussing investments.",
        "Never use finance tools for self-questions.",
    ],
    db=db,
    add_history_to_context=True,
    markdown=True,
)

# -------------------------------------------------
# General Agent
# -------------------------------------------------
general_agent = Agent(
    name="General Agent",
    role="Handles general and self-referential questions",
    model=Ollama(id="llama3.2", host=OLLAMA_HOST),
    instructions=COMMON_INSTRUCTIONS + [
        "Answer from reasoning and internal knowledge.",
        "Do NOT use external tools unless absolutely necessary.",
    ],
    db=db,
    add_history_to_context=True,
    markdown=True,
)

# -------------------------------------------------
# Orchestrator Agent
# -------------------------------------------------
orchestrator_agent = Agent(
    name="Orchestrator Agent",
    role="Coordinates other agents and produces the final answer",
    model=Ollama(id="llama3.2", host=OLLAMA_HOST),
    instructions=[
        "You do not answer directly without coordination.",
        "Decide which agents are needed for the task.",
        "Delegate subtasks to other agents.",
        "Combine their outputs into a single final answer.",
    ],
    db=db,
    markdown=True,
)

def run_a2a_workflow(user_query: str) -> str:
    print("\n=== ORCHESTRATION START ===")
    print("User query:", user_query)

    web_output = ""
    finance_output = ""

    if any(word in user_query.lower() for word in ["news", "latest", "today", "recent"]):
        print("→ Web Agent selected")
        web_output = web_agent.run(
            f"Provide factual, concise information for: {user_query}"
        )
    else:
        print("→ Web Agent skipped")

    if any(ticker in user_query.upper() for ticker in ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN"]):
        print("→ Finance Agent selected")
        finance_output = finance_agent.run(
            f"Use live market data to answer: {user_query}"
        )
    else:
        print("→ Finance Agent skipped")

    print("→ General Agent synthesizing final answer")

    final_answer = general_agent.run(
        f"""
        User Question:
        {user_query}

        Web Agent Response:
        {web_output}

        Finance Agent Response:
        {finance_output}

        Combine everything into a clear final answer.
        """
    )

    print("=== ORCHESTRATION END ===\n")

    return final_answer



# -------------------------------------------------
# AgentOS runtime
# -------------------------------------------------
agent_os = AgentOS(
    agents=[
        web_agent,
        finance_agent,
        general_agent,
    ]
)

app = agent_os.get_app()
# -------------------------------------------------
# A2A request schema
# -------------------------------------------------

class A2ARequest(BaseModel):
    query: str

# -------------------------------------------------
# A2A endpoint
# -------------------------------------------------
@app.post("/a2a")
async def a2a_endpoint(payload: A2ARequest):
    return {
        "response": run_a2a_workflow(payload.query)
    }

# -------------------------------------------------
# Run server
# -------------------------------------------------
if __name__ == "__main__":
    agent_os.serve(app="app:app", reload=True)
