from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

chat_history = [SystemMessage("You are a helpful assistant.")]
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    user_input = HumanMessage(user_input.strip().join(". Answer the question in a concise manner, preferably one word."))
    chat_history.append(user_input)
    result = AIMessage(content = model.invoke(chat_history).content)
    chat_history.append(result.content)
    print(f"Bot: {result}")
