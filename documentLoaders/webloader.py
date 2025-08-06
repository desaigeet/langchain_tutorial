from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
 
url = "https://www.screener.in/company/INFY/consolidated/"
loader = WebBaseLoader(url)
document = loader.load()


model = ChatOpenAI(
    model = "gpt-3.5-turbo"
)

prompt = PromptTemplate(
    template = "Give me 5 reasons why i should iunvest in the following company and 5 reasons why i should not?, {document}",
    input_variables = ["document"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"document": document[0].page_content})
print(result)
