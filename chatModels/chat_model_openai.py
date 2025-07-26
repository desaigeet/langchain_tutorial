from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = "gpt-4")

result = model.invoke("Write a 5 line description on Virat Kohli?", temperature=0.8)

# print(result)
print(result.content)
