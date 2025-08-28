import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from registry import getLlm
from dotenv import load_dotenv
from AgentsGraph import AgentsGraph
import pprint


if __name__ == "__main__":
    ag = AgentsGraph()
    graph = ag.get_graph()
    while 1==1:
        question = input("Enter your question (-1 to quit) : ")
        if(question == "-1"):
            break
        inputs = {
        "messages": [
            ("user", question),
        ]
        }
        for event in graph.stream(inputs):
            for key, value in event.items():
                if value is None:
                    continue
                pprint.pprint(f"Output from node '{key}':")
                pprint.pprint(value, indent=2, width=80, depth=None)
                print()
        # result = graph.invoke(inputs)
        # print("Answer: " +result["messages"][-1].content)
        # print("Source :" + result['sources'])
        print("\n")
