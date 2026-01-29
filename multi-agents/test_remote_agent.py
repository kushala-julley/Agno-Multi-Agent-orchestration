import asyncio
from agno.agent import RemoteAgent

async def main():
    finance_agent = RemoteAgent(
        base_url="http://localhost:7777/a2a/agents/finance-agent",
        agent_id="finance-agent",
        protocol="a2a",
    )

    response = await finance_agent.arun(
        "What is the current stock price of AAPL?"
    )

    print(response)

if __name__ == "__main__":
    asyncio.run(main())
