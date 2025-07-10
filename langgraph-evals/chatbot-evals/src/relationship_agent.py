from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# Define input schema
class InputState(TypedDict):
    question: str
# Define output schema
class OutputState(TypedDict):
    answer: str
# Combine input and output
class OverallState(InputState, OutputState):
    pass

def chat_node(state: InputState):
    question =  state["question"]
    # Define system and user prompts
    # system_message = SystemMessage(content="You are a relationship expert. You are given a question and you need to answer it. You only answer in 20 words.")
    system_message = SystemMessage(content="You are a relationship expert. You are given a question and you need to answer it. You only answer in 10 words. You must keep the answer short and concise as much as possible.")
    user_message = HumanMessage(content=question)
    
    answer = llm.invoke([system_message, user_message])
    return {"answer": answer.content, "question": state["question"]}


graph_builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
graph_builder.add_node("chat", chat_node)
graph_builder.set_entry_point("chat")
graph_builder.set_finish_point("chat")
graph = graph_builder.compile()