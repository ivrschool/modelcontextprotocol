
from langgraph.graph import StateGraph, START, END, MessagesState
from typing_extensions import TypedDict

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
    system_prompt = """
                    You are a relationship expert. You are given a question and you need to answer it.
                    """
    prompt = f"""
                {system_prompt}
                {question}
              """
    
    answer = llm.invoke(prompt)
    return {"answer": answer.content, "question": state["question"]}

graph_builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)

graph_builder.add_node("chat", chat_node)

graph_builder.set_entry_point("chat")
graph_builder.set_finish_point("chat")

relationship_graph = graph_builder.compile()
