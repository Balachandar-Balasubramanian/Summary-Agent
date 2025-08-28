from Agents.SupervisorAgent import supervisor_node
from Agents.SummaryAgent import summary_node
from Agents.KnowledgeAgent import knowledgeBase_node
from Agents.TextStylerAgent import Text_styler_node
from registry import getLlm
from pydantic import BaseModel
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState


from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    history: Annotated[list, add_messages]
    message_type: str | None
    


class AgentsGraph:
    def __init__(self):
        graph = StateGraph(MessagesState)
        graph.add_node("supervisor", supervisor_node) 
        graph.add_node("Knowledge_Agent", knowledgeBase_node)  
        graph.add_node("Summary_Agent", summary_node)
        graph.add_node("Chat_Styler", Text_styler_node) 
 

        graph.add_edge(START, "supervisor")
        self.app = graph.compile()

    def get_graph(self):
        return self.app


