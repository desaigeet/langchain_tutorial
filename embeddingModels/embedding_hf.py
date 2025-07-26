from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

document_embedding = False
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
    print(np.array(result).shape)
else:
    result = embedding.embed_query("What is the capital of India?")
    print(result)
    print(np.array(result).shape)
