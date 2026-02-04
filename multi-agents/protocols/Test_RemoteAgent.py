from agno.agent import RemoteAgent
import asyncio

async def main():
    finance_agent = RemoteAgent(
        base_url="http://localhost:7777/a2a/agents/finance-agent",  # ✅ SERVER ROOT ONLY
        agent_id="finance-agent",              # ✅ REQUIRED
        protocol="a2a",
        timeout=300,
    )

    response = await finance_agent.arun(
        "What is the current stock price of AAPL?"
    )

    print("\nFINAL RESPONSE:\n", response)

asyncio.run(main())
