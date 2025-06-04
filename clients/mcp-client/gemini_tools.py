from google.genai.types import Tool, FunctionDeclaration
from schema_utils import clean_schema


def convert_mcp_tools_to_gemini(mcp_tools):
    """
    Converts MCP tool definitions to the Gemini-compatible format for function calling.

    Args:
        mcp_tools (list): List of MCP tool objects with 'name', 'description', and 'inputSchema'.

    Returns:
        list: List of Gemini Tool objects with properly formatted function declarations.
    """
    gemini_tools = []

    for tool in mcp_tools:
        # Ensure inputSchema is valid and clean
        parameters = clean_schema(tool.inputSchema)

        # Construct Gemini-compatible function declaration
        function_declaration = FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=parameters
        )

        # Wrap in Tool object
        gemini_tool = Tool(function_declarations=[function_declaration])
        gemini_tools.append(gemini_tool)

    return gemini_tools
