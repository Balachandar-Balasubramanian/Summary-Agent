from langchain_community.tools.tavily_search import TavilySearchResults 
import os
from langchain.tools import Tool
from dotenv import load_dotenv


tavily_search = TavilySearchResults(max_results=2)


