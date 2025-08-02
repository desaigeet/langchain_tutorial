from langchain_huggingface import ChatHuggingFace  , HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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


template1 = PromptTemplate(
    template="Write a detailed explanation on: {topic}",
    input_variables=['topic']
)
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text: \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()
# prompt1 = template1.invoke({"topic":"LangChain"})
# result = model.invoke(prompt1)
# prompt2 = template2.invoke({"text": result.content})
# result = model.invoke(prompt2)

# print(result.content)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "LangChain"})

print(result)
