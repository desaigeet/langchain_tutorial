from langchain_core.tools import tool 
from langchain_openai import ChatOpenAI
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
load_dotenv()

search_tool = DuckDuckGoSearchRun()
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm = llm,
    tools = [search_tool], 
    prompt = prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
    handle_parsing_errors=True
)

response = agent_executor.invoke({
    "input" : "What are the impact of Trump tariffs on the American consumer?"
})

print(response)
