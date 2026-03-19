import os

from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch



# @tool
# def search(query: str) -> dict: 
#     """Tool to search over the internet
#     Args: 
#         query: The search query to execute
#     Returns: The search results
#     """

#     print(f"Searching for: {query}")
#     return tavily_client.search(query=query)


llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model = llm, tools = tools)

def main():
    # print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for 3 job postings of a Software Engineer using Node JS in Bengaluru on linkedin and list their details")]})
    print(result)

if __name__ == "__main__":
    main()
