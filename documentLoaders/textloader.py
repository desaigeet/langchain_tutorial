from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

loader = TextLoader(
    "data/cricket_overview.txt",
    encoding="utf-8")

documents = loader.load()

print("Length of Document:", len(documents))

model = ChatOpenAI(
    model = "gpt-3.5-turbo"
)

prompt = PromptTemplate(
    template = "Write a poem from the following text: {text}",
    input_variables= ["text"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"text": documents[0].page_content})

print("Poem Result:", result)
