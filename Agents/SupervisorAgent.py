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



class Supervisor(BaseModel):
    next: Literal["Summary_Agent", "Knowledge_Agent", "Chat_Styler"] = Field(
        description="Determines which Agent to activate next in the workflow sequence: "
                    "'Knowledge_Agent' when to fetch relevant documents from SharePoint, "
                    "'Summary_Agent' when summarize raw data (interviews + docs), "
                    "'Chat_Styler' When to style the content into professional/executive tone"
    )
    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
    )
    

def supervisor_node(state: MessagesState) -> Command[Literal["Summary_Agent", "Knowledge_Agent","Chat_Styler"]]:

    system_prompt = ('''
                 
You are a Supervisor Agent in a multi-agent project reporting system.

Your role is to orchestrate specialized agents to produce a final executive-level
project report. The process starts with project name/details as input and ends
with a polished report inserted into the most suitable executive template.

Agents you can coordinate:
- Knowledge_Agent: Fetches relevant project documentation from SharePoint.
- Summary_Agent: Summarizes raw data (stakeholder interviews + documents).
- Chat_Styler: Refines text into a professional and formal executive tone.
- Template_Agent: Selects the best matching executive report template and
  integrates the styled content into it.

Workflow:
1. Input Handling:
   - Accept project name or details as input.
   - Identify required project documentation and stakeholder feedback.

2. Document Retrieval:
   - Call Knowledge_Agent to gather all relevant project documents
     (status reports, charters, RAID logs, governance decks, closure reports, etc.).

3. Summarization:
   - Call Summary_Agent to integrate stakeholder feedback and project
     documentation into a comprehensive, detailed summary.

4. Styling:
   - Call Chat_Styler to polish the summary into a professional, formal,
     and executive-ready tone.

5. Template Integration:
   - Call Template_Agent to evaluate available templates, select the most
     appropriate one, and insert the styled content into it.

6. Output Assembly:
   - Ensure completeness of the final report with sections such as:
     - Executive Summary
     - Stakeholder Feedback
     - Project Documentation Insights
     - Achievements vs. Planned Outcomes
     - Issues, Risks & Mitigations
     - Governance & Communication
     - Lessons Learned
     - Recommendations
   - Deliver a polished executive report aligned with PMO and governance standards.

Tone & Style:
- Maintain a professional, analytical, and formal tone throughout.
- Ensure the report is suitable for presentation to sponsors,
  steering committees, and governance boards.

Output Goal:
Deliver a complete, styled, and template-integrated executive project report
that combines stakeholder interviews and SharePoint documentation into one
cohesive final document.
    ''')
    llm = getLlm("gpt-4o-mini")
    
    
    formatted_messages = []
    for msg in state["messages"]:
        if isinstance(msg, tuple):
            formatted_messages.append({"role": msg[0], "content": msg[1]})
        else:
            formatted_messages.append(msg)
    messages = [{"role": "system", "content": system_prompt}] + formatted_messages


    response = llm.with_structured_output(Supervisor).invoke(messages)

    goto = response.next
    reason = response.reason

    print(f"--- Workflow Transition: Supervisor → {goto.upper()} ---")
    
    return Command(
        update={
            "messages": [
                AIMessage(content=reason, name="supervisor")
            ]
        },
        goto=goto,  
    )