from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.playground import Playground, serve_playground_app

# ---------------------------
# Flowchart Agent (Local Ollama)
# ---------------------------
flowchart_agent = Agent(
    name="flowchart_agent",
    role="Get flowchart information and explain processes step by step",
    model=Ollama(model="llama3.2"),
    tools=[DuckDuckGoTools()],
    instructions=[
        "Explain concepts clearly",
        "When appropriate, describe flowcharts step by step"
    ],
    markdown=True,
)

# ---------------------------
# Playground App
# ---------------------------
app = Playground(
    agents=[flowchart_agent]
).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
