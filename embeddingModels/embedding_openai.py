from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=64)

document_embedding = True
questions = [
    "What is the capital of India?",
    "Who is the president of the United States?",
    "What is the tallest mountain in the world?",
    "How does photosynthesis work?",
    "What is the population of China?",
    "Explain the theory of relativity.",
    "What are the benefits of exercise?",
    "Who wrote 'To Kill a Mockingbird'?",
    "What is quantum computing?",
    "How do vaccines work?"
]

if document_embedding:
    result = embedding.embed_documents(questions)
    print(result)
    print(type(result)) 
else:
    result = embedding.embed_query("What is the capital of India?")
    print(result)
    print(type(result))
