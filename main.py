import os

from dotenv import load_dotenv
load_dotenv()
from typing import List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

# @tool
# def search(query: str) -> dict :
#     """Search the internet for real-time, current information.
#     Args:
#         query: The search query to execute
#     Returns: The search results as text
#     """
    # return {"answer": "The current weather in Tokyo, Japan is 200 degrees Celsius"}
    # return {"answer": "The population of China is 200000"}


class Source(BaseModel):
    """Schema for a source used by the agent to find information""" 
    url:str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources""" 
    answer:str = Field(description="The agent's response to the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


llm = ChatOpenAI()
tools = [TavilySearch()]
# tools = [search]
agent = create_agent(model = llm, tools = tools, response_format=AgentResponse)

def main():
    # print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for 3 job postings of a Software Engineer using Node JS in Bengaluru on linkedin and list their details")]})
    # result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
    # result = agent.invoke({"messages": [HumanMessage(content="What is the population of China?")]})
    print(result)

if __name__ == "__main__":
    main()
