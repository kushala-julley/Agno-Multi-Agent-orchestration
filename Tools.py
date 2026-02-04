from agno.agent import Agent
from agno.tools import tool
from agno.models.ollama import Ollama

@tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: 72°F, sunny"

agent = Agent(
    model=Ollama(id="llama3.2", host="http://localhost:11434"),
    tools=[get_weather]
)

agent.print_response("What's the weather in San Francisco?")
