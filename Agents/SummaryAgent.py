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
from Tools import WebSearchTool


def summary_node(state: MessagesState) -> Command[Literal["supervisor"]]:

    """
Enterprise_Project_Review_Summarizer.
Takes interview notes and project documentation (e.g., from SharePoint),
and generates a comprehensive, structured project review report
covering stakeholder feedback, achievements, risks, lessons learned,
and recommendations for future projects
    """
    
    research_agent = create_react_agent(
        getLlm("gpt-4o-mini"),  
        tools=[],  
        state_modifier= """
You are an Enterprise Project Review Summarizer.

Your task is to take:
- Interview notes (questions and answers from project sponsors, steering committee, key business users, project managers, and project team members).
- Project documentation and pages.

Then produce a very detailed, structured project review summary.

Instructions:
1. Preserve Completeness:
   - Do not omit any significant detail, risk, issue, or lesson learned.
   - If feedback contradicts, capture all perspectives instead of merging them.

2. Organize by Category:
   Structure the output as a formal project review report, including:
   - Executive Summary
   - Sponsor & Steering Committee Feedback
   - Business User Feedback
   - Project Management Feedback
   - Project Team Feedback
   - Key Findings from SharePoint / Documentation
   - Achievements vs. Planned Outcomes
   - Issues, Risks & Mitigations
   - Dependencies & Change Requests
   - Governance, Communication & Collaboration
   - Lessons Learned
   - Recommendations for Future Projects
   - Appendices (raw extracts from interviews and documents)

3. Depth of Analysis:
   - Provide lengthy, exhaustive summaries — not short overviews.
   - Expand on each point with supporting context from interviews or documents.
   - If interviewees highlight challenges, detail the root causes, impact, and outcomes.
   - If documents provide metrics (timelines, budgets, deliverables), integrate them into the narrative.

4. Tone & Style:
   - Formal, neutral, and analytical.
   - Suitable for a PMO or steering committee audience.
   - Clearly distinguish facts (from documents) vs. opinions (from stakeholders).

5. Output Goal:
   - Deliver a comprehensive project closure-style report that could be directly used for governance, audits, or lessons learned repositories.
"""
    )

    result = research_agent.invoke(state)

    print(f"--- Workflow Transition: Researcher → Supervisor ---")

    return Command(
        update={
            "messages": [ 
                HumanMessage(
                    content=result["messages"][-1].content,  
                    name="Summary_Agent"  
                )
            ]
        },
        goto="supervisor", 
    )

