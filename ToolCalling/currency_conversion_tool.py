from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
import requests
import json
from dotenv import load_dotenv

load_dotenv()

@tool
def fetch_conversion_factor(base_currency: str, conversion_currency: str) -> float:
    """
    Fetches the conversion rate from exchange rate api by taking into account the base_Cuurent and conversion_currency
    Expects the currency codes as input
    """

    api = "a9bc9a67c9b51d91abe530f0"
    url = f"https://v6.exchangerate-api.com/v6/{api}/pair/{base_currency}/{conversion_currency}/"

    response = requests.get(url)
    data = response.json()

    return data

@tool
def convert_curency(amount: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    Converts the amount from base_currency to conversion_currency using the conversion rate
    """
    return amount * conversion_rate

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_tool = llm.bind_tools([fetch_conversion_factor, convert_curency])

query = HumanMessage("What is the conversion factor between USD and EUR. Also convert 100 USD to EUR.")
messages = [query]

result = llm_tool.invoke(messages)
messages.append(result)


conversion_tool_call = result.tool_calls[0]
convertor_tool_call = result.tool_calls[1]

conversion_rate = fetch_conversion_factor.invoke(conversion_tool_call)
messages.append(conversion_rate)
conversion_rate = json.loads(conversion_rate.content)["conversion_rate"]

convertor_tool_call['args']['conversion_rate'] = conversion_rate
converted_amount = convert_curency.invoke(convertor_tool_call)
messages.append(converted_amount)
print(messages)

print(llm_tool.invoke(messages).content)
