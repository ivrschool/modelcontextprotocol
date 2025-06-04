import sys
import asyncio
from mcp_client import MCPClient


def print_usage():
    """Prints usage instruction if the script is run without the server path."""
    print("Usage: python client.py <path_to_server_script>")


async def main():
    """Main function to start the MCP client, connect to the server, and begin the chat loop."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())