import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


llm_gpt4o_mini = ChatOpenAI(model="gpt-4o-mini",api_key=openai_api_key)
llm_gpt3_5_turbo = ChatOpenAI(model="gpt-3.5-turbo",api_key=openai_api_key)

def getLlm(model: str = "gpt-4o-mini"):
    if (model=="gpt-4o-mini"):
        return llm_gpt4o_mini
    elif(model == "gpt-3.5-turbo"):
        return llm_gpt3_5_turbo
    else:
        return llm_gpt4o_mini