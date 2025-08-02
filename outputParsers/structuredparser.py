from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, OutputFixingParser
from transformers import pipeline

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    task = "text-generation",
    repo_id="google/gemma-2-2b-it",
)
model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='mathematics', description='Marks in Mathematics as integer'),
    ResponseSchema(name='science', description='Marks in Science as integer'),
    ResponseSchema(name='english', description='Marks in English as integer'),
    ResponseSchema(name='name', description='Name of the student'),
    ResponseSchema(name='age', description='Age of the student as integer'),
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="""
    Give name, age and marks of a fictional student.  \n  {format_instruction}""",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)
