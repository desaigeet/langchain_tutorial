from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.7)

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("What is the capital of France?")
]
results = model.invoke(messages)

print(AIMessage(content=results.content))

print(messages)
