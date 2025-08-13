from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv
load_dotenv()

@tool
def multiply(x: int, y: int) -> int:
    """
    Multiply two numbers
    """
    return x * y

llm = ChatOpenAI(model = "gpt-3.5-turbo")
llm_tool = llm.bind_tools([multiply])

query = HumanMessage("Hey, Can you multiply 2 and 1000?")
messages = [query]
result = llm_tool.invoke(messages)
messages.append(result)
tool_call = result.tool_calls[0]
tool_result = multiply.invoke(tool_call) 
messages.append(tool_result)

print(llm_tool.invoke(messages).content)
