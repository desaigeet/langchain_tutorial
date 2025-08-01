from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

chat_history = []
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    user_input = user_input.strip().join(". Answer the question in a concise manner, preferably one word.")
    chat_history.append(user_input)
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print(f"Bot: {result.content}")
