import asyncio
from fastmcp import Client

async def call_validate():
    async with Client("https://mighty-games-invent.loca.lt/mcp/", auth="1e3aa21879b4") as client:
        result = await client.call_tool("notion_summarizer", {
            "input": {
                "url": "https://www.notion.so/Foodery-f3d154a0015647f9993321d851104080?source=copy_link"
            }
        })
        print(result)

asyncio.run(call_validate())
