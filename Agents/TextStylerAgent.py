from typing import Annotated, Sequence, List, Literal 
from pydantic import BaseModel, Field 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults 
from langgraph.types import Command 
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent 
from IPython.display import Image, display 
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from registry import getLlm

def Text_styler_node(state: MessagesState) -> Command[Literal["__end__"]]:

    """
    Chat_Styler Node Transforms raw text into a polished, professional, and formal tone, ready for inclusion in an executive-level project report..
    """
    
    research_agent = create_react_agent(
        getLlm("gpt-4o-mini"),  
        tools=[],  
        state_modifier= """
                    You are a TextStyler Agent.

Your role is to refine and rewrite given text into a professional,
formal, and executive-level tone suitable for inclusion in a final
project report.

Instructions:
1. Maintain Meaning:
   - Preserve the original intent, facts, and details.
   - Do not omit or add new information.

2. Tone & Style:
   - Use clear, precise, and formal business language.
   - Avoid slang, casual phrases, or conversational wording.
   - Ensure the style aligns with executive reporting standards.

3. Structure & Readability:
   - Improve flow and coherence of sentences.
   - Use concise phrasing while retaining completeness.
   - Ensure grammar, punctuation, and formatting are correct.

4. Output Goal:
   - Provide polished text that is ready to be directly placed
     into an executive report for sponsors, committees, and PMOs.
                     """
    )

    result = research_agent.invoke(state)

    print(f"--- Workflow Transition: Chat_Styler → FINISH ---")

    return Command(
        update={
            "messages": [ 
                AIMessage(
                    content=result["messages"][-1].content,  
                    name="Chat_Styler"  
                )
            ]
        },
        goto="FINISH", 
    )