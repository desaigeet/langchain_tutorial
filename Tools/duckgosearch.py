from langchain_community.tools import DuckDuckGoSearchRun

search_tools = DuckDuckGoSearchRun()

results = search_tools.invoke("Trump Tariff News")

print(results)
