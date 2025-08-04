from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template = "Generate a detailed explanation about {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Generate a 5 pointer summary of the following explanation: {explanation}",
    input_variables=["explanation"]
)

parser = StrOutputParser()

model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({
    "topic": "Langchain"
})

print(result)

chain.get_graph().print_ascii()
