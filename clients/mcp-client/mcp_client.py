import os
from typing import Optional
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from google.genai import types
from gemini_tools import convert_mcp_tools_to_gemini
from gemini_config import configure_gemini


class MCPClient:
    def __init__(self):
        """Initialize the MCP client, load Gemini API key, and set up client object."""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.genai_client = configure_gemini()

    async def connect_to_server(self, server_script_path: str):
        """
        Connect to the MCP server, initialize the session, and fetch available tools.

        Args:
            server_script_path (str): Path to the MCP server script.
        """
        command = "python" if server_script_path.endswith('.py') else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        self.function_declarations = convert_mcp_tools_to_gemini(tools)

    async def process_query(self, query: str) -> str:
        """
        Process a user query with Gemini and optionally execute tool calls.

        Args:
            query (str): The user's input query.

        Returns:
            str: Formatted response from Gemini.
        """
        user_prompt_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=query)]
        )

        response = self.genai_client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=[user_prompt_content],
            config=types.GenerateContentConfig(tools=self.function_declarations),
        )

        final_text = []

        for candidate in response.candidates:
            if candidate.content.parts:
                for part in candidate.content.parts:
                    if isinstance(part, types.Part):
                        if part.function_call:
                            function_call_part = part
                            tool_name = function_call_part.function_call.name
                            tool_args = function_call_part.function_call.args

                            print(f"\n[Gemini requested tool call: {tool_name} with args {tool_args}]")

                            try:
                                result = await self.session.call_tool(tool_name, tool_args)
                                function_response = {"result": result.content}
                            except Exception as e:
                                function_response = {"error": str(e)}

                            function_response_part = types.Part.from_function_response(
                                name=tool_name,
                                response=function_response
                            )

                            function_response_content = types.Content(
                                role='tool',
                                parts=[function_response_part]
                            )

                            response = self.genai_client.models.generate_content(
                                model='gemini-2.0-flash-001',
                                contents=[
                                    user_prompt_content,
                                    function_call_part,
                                    function_response_content,
                                ],
                                config=types.GenerateContentConfig(tools=self.function_declarations),
                            )

                            text_part = response.candidates[0].content.parts[0].text
                            if text_part:
                                final_text.append(text_part)

                        else:
                            final_text.append(part.text)

        return "\n".join(part for part in final_text if part is not None)


    async def chat_loop(self):
        """Run an interactive chat session with the user."""
        print("\nMCP Client Started! Type 'quit' to exit.")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
            response = await self.process_query(query)
            print("\n" + response)

    async def cleanup(self):
        """Clean up async resources before exiting."""
        await self.exit_stack.aclose()
