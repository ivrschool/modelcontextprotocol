#!/usr/bin/env python

import asyncio
import os
import sys
from typing import Optional

from mcp import ClientSession
from mcp.client.sse import sse_client

from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration
from google.genai.types import GenerateContentConfig

from dotenv import load_dotenv

load_dotenv()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None
        self.chat_history_summary = ""  # <-- This is the memory (summary of conversation so far)

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file.")

        self.genai_client = genai.Client(api_key=gemini_api_key)

    async def connect_to_sse_server(self, server_url: str):
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        await self.session.initialize()

        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        self.function_declarations = convert_mcp_tools_to_gemini(tools)

    async def cleanup(self):
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, query: str) -> str:
        contents = []

        if self.chat_history_summary:
            memory_part = types.Content(
                role='user',
                parts=[types.Part.from_text(text=f"This is the summary of our conversation so far: {self.chat_history_summary}")]
            )
            contents.append(memory_part)

        user_prompt_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=query)]
        )
        contents.append(user_prompt_content)

        response = self.genai_client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=contents,
            config=types.GenerateContentConfig(
                tools=self.function_declarations,
            ),
        )

        final_text = []

        for candidate in response.candidates:
            if candidate.content.parts:
                for part in candidate.content.parts:
                    if part.function_call:
                        tool_name = part.function_call.name
                        tool_args = part.function_call.args
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
                                *contents,
                                part,
                                function_response_content,
                            ],
                            config=types.GenerateContentConfig(
                                tools=self.function_declarations,
                            ),
                        )

                        final_text.append(response.candidates[0].content.parts[0].text)
                    else:
                        final_text.append(part.text)

        # After getting full response, update chat history summary
        response_text = "\n".join(final_text)
        self.update_chat_history_summary(query, response_text)

        return response_text

    def update_chat_history_summary(self, user_query: str, model_response: str):
        # Very simple memory strategy: append last turn as 1 sentence
        # In production, you'd want a summarizer here
        new_summary_entry = f"User asked: '{user_query}'. Assistant replied: '{model_response}'. "
        self.chat_history_summary += new_summary_entry

    async def chat_loop(self):
        print("\nMCP Client Started! Type 'quit' to exit.")

        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break

            response = await self.process_query(query)
            print("\n" + response)

def clean_schema(schema):
    if isinstance(schema, dict):
        schema.pop("title", None)
        if "properties" in schema and isinstance(schema["properties"], dict):
            for key in schema["properties"]:
                schema["properties"][key] = clean_schema(schema["properties"][key])
    return schema

def convert_mcp_tools_to_gemini(mcp_tools):
    gemini_tools = []

    for tool in mcp_tools:
        parameters = clean_schema(tool.inputSchema)

        function_declaration = FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=parameters
        )

        gemini_tool = Tool(function_declarations=[function_declaration])
        gemini_tools.append(gemini_tool)

    return gemini_tools

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client_sse.py <server_url>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_sse_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
