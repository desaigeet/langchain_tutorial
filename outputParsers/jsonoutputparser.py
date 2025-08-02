from langchain_huggingface import ChatHuggingFace  , HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from transformers import pipeline

from dotenv import load_dotenv

load_dotenv()

model = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # max_length=512,
)

llm = HuggingFacePipeline(pipeline=model)
model = ChatHuggingFace(llm=llm)


parser = JsonOutputParser()

template = PromptTemplate(
    template="Give name, age and net worth of a fictional persion \n  {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)
