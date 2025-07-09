# general_agent_graph.py

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode

load_dotenv()

class GeneralAgentGraph:
    def __init__(self, tools):
        self.tools = tools
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0).bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        self.graph = self._build_graph()

    def call_model(self, state: MessagesState):
        messages = state["messages"]
        response = self.model.invoke(messages)
        return {"messages": response}

    def should_continue(self, state: MessagesState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    def _build_graph(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {"tools": "tools", "__end__": END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def get_graph(self):
        return self.graph
