from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

model = ChatOpenAI(
    model = "gpt-3.5-turbo",
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Generate a joke on {topic}",
    input_variables=["topic"]
)

# prompt2 = PromptTemplate(
#     template = "Explain the joke {joke}.",
#     input_variables=["joke"] 
# )

chain = RunnableSequence(prompt1, model, parser)

print(chain.invoke({"topic": "Narendra Modi"}))
