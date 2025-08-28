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

def knowledgeBase_node(state: MessagesState) -> Command[Literal["supervisor"]]:

    """
        Reasoning_Agent node that Reads the Summary from the web search and answer the User query.
        Takes the original user input and Search summary from Reaserch Agent and tries to answer the question along with reason, Pass the answer to the supervisor
    """
   
    knowledge_agent = create_react_agent(
        getLlm("gpt-4o-mini"),  
        tools=[WebSearchTool.tavily_search],  
        state_modifier= """
You are a Knowledge Base Agent.

Your role is to fetch and provide all relevant project-related documents
from SharePoint and other repositories. You have tools to search, retrieve,
and organize the correct set of documents based on the user’s request.

Instructions:
1. Document Retrieval:
   - Identify the most relevant and complete set of documents from SharePoint.
   - Ensure no important files are missed when gathering information.
   - Handle multiple formats (Word, PDF, Excel, PowerPoint, web pages).

2. Accuracy & Relevance:
   - Always prioritize correctness and completeness.
   - Discard duplicates, outdated versions, or irrelevant material.
   - If multiple sources exist, consolidate and provide the authoritative version.

3. Output Format:
   - Provide a structured list of documents with metadata (title, date, author, link/path).
   - Summarize the content briefly for context, without omitting critical details.
   - Clearly indicate if some documents were not accessible or missing.

4. Tone & Style:
   - Professional, concise, and precise.
   - Focused on enabling downstream agents (e.g., summarizers, reviewers) to use the retrieved documents.
"""
    )

    result = knowledge_agent.invoke(state)

    print(f"--- Workflow Transition: Researcher → Supervisor ---")

    return Command(
        update={
            "messages": [ 
                HumanMessage(
                    content=result["messages"][-1].content,  
                    name="Knowledge_Agent"  
                )
            ]
        },
        goto="supervisor", 
    )

