import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel

# -------------------------------------------------
# Monkey-patch to fix Agno's nested JSON parsing bug
# -------------------------------------------------
import ast
import agno.utils.functions as agno_functions

_original_get_function_call = agno_functions.get_function_call

def _parse_nested_value(v):
    """Try to parse a string that looks like a Python/JSON list or dict."""
    if not isinstance(v, str):
        return v

    stripped = v.strip()
    if not stripped.startswith(('[', '{')):
        return v

    # Try JSON first (double quotes)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try Python literal (single quotes) - handles ['hobbies'] syntax
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        pass

    return v  # Return original if all parsing fails

def _patched_get_function_call(name, arguments=None, call_id=None, functions=None):
    """Patched version that handles nested JSON/Python strings like topics: '["a"]' or "['a']" """
    if arguments is not None and arguments != "":
        try:
            _args = json.loads(arguments)
            if isinstance(_args, dict):
                for k, v in _args.items():
                    _args[k] = _parse_nested_value(v)
                arguments = json.dumps(_args)
        except (json.JSONDecodeError, ValueError):
            pass  # Let original function handle errors

    return _original_get_function_call(name, arguments, call_id, functions)

agno_functions.get_function_call = _patched_get_function_call
# -------------------------------------------------

# AgentOS A2A Interface
from agno.os.interfaces.a2a import A2A
# AgentOS
from agno.os import AgentOS
from agno.agent import Agent
# AgentOS DB
from agno.db.sqlite import SqliteDb
# AgentOS Models
from agno.models.ollama import Ollama
# AgentOS Tools
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
    update_memory_on_run=True,
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
    update_memory_on_run=True,
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
    update_memory_on_run=True,
)


a2a = A2A(agents=[web_agent, finance_agent, general_agent])



# -------------------------------------------------
# AgentOS runtime
# -------------------------------------------------
agent_os = AgentOS(
    agents=[
        web_agent,
        finance_agent,
        general_agent,
    ],
    interfaces=[a2a],
    db=db,
    tracing=True,
)

app = agent_os.get_app()
if __name__ == "__main__":
    agent_os.serve(app="app:app", reload=True)
