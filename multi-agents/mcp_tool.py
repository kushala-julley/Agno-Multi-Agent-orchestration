import asyncio
from agno.agent import Agent
from agno.tools.mcp import MCPTools
from agno.models.ollama import Ollama

async def main():
    async with MCPTools(
        "npx -y @modelcontextprotocol/server-filesystem .",
        include_tools=["search_files"],  # restrict tools
    ) as mcp:
        agent = Agent(
            model=Ollama(id="llama3.2"),
            tools=[mcp],
            instructions=(
                "You are a filesystem assistant.\n"
                "Use the MCP search_files tool.\n"
                "Call it with query='*.py' and path='.'.\n"
                "Do NOT suggest Python code or workarounds.\n"
                "Return the file list."
            ),
            markdown=True,
        )

        await agent.aprint_response(
            "List all Python files in this directory",
            stream=True,
        )

asyncio.run(main())
