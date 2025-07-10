import os
import ast
import asyncio
from dotenv import load_dotenv
from langsmith import Client, wrappers
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage
import openai
import pytest

# =====================
# CONFIG & CONSTANTS
# =====================
load_dotenv()
DATASET_NAME = "Chatbot QA Dataset"
MCP_URL = "http://localhost:2024/mcp"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"

# =====================
# TOOL SETUP
# =====================
def get_tools():
    async def mcp_tools_node():
        client = MultiServerMCPClient({
            "agent": {
                "transport": "streamable_http",
                "url": MCP_URL,
            }
        })
        tools = await client.get_tools()
        return tools
    return asyncio.run(mcp_tools_node())

tools = get_tools()
tool_node = ToolNode(tools)

# =====================
# DATASET SETUP
# =====================
client = Client()

def get_or_create_dataset():
    try:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME, description="A test dataset for relationship agent."
        )
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"question": "How do I rebuild trust after a breakup?"},
                    "outputs": {"answer": "Rebuilding trust takes time, honest communication, and consistent actions."},
                },
                {
                    "inputs": {"question": "What should I do if my partner and I keep arguing?"},
                    "outputs": {"answer": "Try to identify the root cause and communicate calmly and openly."},
                },
                {
                    "inputs": {"question": "Is it normal to have doubts in a long-term relationship?"},
                    "outputs": {"answer": "Yes, occasional doubts are common and worth discussing respectfully."},
                },
                {
                    "inputs": {"question": "How can I improve communication with my partner?"},
                    "outputs": {"answer": "Practice active listening, be honest, and avoid blame during conversations."},
                },
                {
                    "inputs": {"question": "What are signs of a healthy relationship?"},
                    "outputs": {"answer": "Trust, respect, open communication, and emotional support are key signs."},
                }
            ]
        )
    except Exception:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print("dataset already exists. loaded it...")
    return dataset

dataset = get_or_create_dataset()

# =====================
# LLM & JUDGE SETUP
# =====================
Gemini = openai.OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
llm_judge = wrappers.wrap_openai(Gemini)
eval_instructions = "You are an expert evaluating answers to questions."

# =====================
# EVALUATION FUNCTIONS
# =====================
async def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    user_content = f"""Question: {inputs['question']}\nExpected Answer: {reference_outputs['answer']}\nPredicted Answer: {outputs['response']}\nIs the predicted answer correct? Respond with CORRECT or INCORRECT."""
    response = llm_judge.chat.completions.create(
        model="gemini-2.5-flash",
        temperature=0,
        messages=[
            {"role": "system", "content": eval_instructions},
            {"role": "user", "content": user_content},
        ],
    ).choices[0].message.content
    return response.strip() == "CORRECT"

async def concision(outputs: dict, reference_outputs: dict) -> bool:
    return len(outputs["response"]) < 2 * len(reference_outputs["answer"])

# =====================
# TARGET FUNCTION
# =====================
async def ls_target(inputs: dict) -> dict:
    AI_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "relationship_agent",
                "args": {"question": f"{inputs['question']}"},
                "id": "tool_call_id_1",
                "type": "tool_call_1",
            }
        ]
    )
    result = await tool_node.ainvoke({"messages": [AI_message]})
    return {"response": ast.literal_eval(result["messages"][-1].content)["answer"]}

# =====================
# TEST FUNCTION
# =====================
@pytest.mark.asyncio
async def test_concision_score():
    experiment_results = await client.aevaluate(
        ls_target,
        data=DATASET_NAME,
        evaluators=[concision, correctness]
    )
    feedback = client.list_feedback(
        run_ids=[r.id for r in client.list_runs(project_name=experiment_results.experiment_name)],
        feedback_key="concision"
    )
    scores = [f.score for f in feedback]
    assert sum(scores) / len(scores) >= 0.8, "Concision score below 80%"

# =====================
# MAIN (for manual run)
# =====================
if __name__ == "__main__":
    asyncio.run(test_concision_score())