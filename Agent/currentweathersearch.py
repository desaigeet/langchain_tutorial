from langchain_core.tools import tool 
from langchain_openai import ChatOpenAI
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city.
    """

    url = "https://api.weatherstack.com/current?access_key=359a780c0829cb2b25faa7b3b4bcc889"
    querystring = {"query":city}
    response = requests.get(url, params=querystring)

    return response.json()

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm = llm,
    tools = [search_tool, weather_data], 
    prompt = prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, weather_data],
    verbose=True,
    handle_parsing_errors=True
)

response = agent_executor.invoke({
    "input" : "What is the weather of the largest district of Maharashtra, India?"
})

print(response)
