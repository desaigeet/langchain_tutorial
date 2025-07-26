from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
import os
from transformers import pipeline

from dotenv import load_dotenv

load_dotenv()

model = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # max_length=512,
)

llm = HuggingFacePipeline(pipeline=model)

# model = ChatHuggingFace(llm=llm)

result = llm.invoke("What is the Capital of India.")

print(result)
