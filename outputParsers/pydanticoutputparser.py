from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from transformers import pipeline
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

# model = pipeline(
#     "text-generation",
#     model="google/gemma-2-2b-it",
# )

# llm = HuggingFacePipeline(pipeline=model)
llm = HuggingFaceEndpoint(
    task = "text-generation",
    repo_id="google/gemma-2-2b-it",
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person", gt=18)
    city: str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = " Genrate the name, age and city of a fictional person from {country} \n {format_instructions}",
    input_variables=['country'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"country": "India"})
print(result)
