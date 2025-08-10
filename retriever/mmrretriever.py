from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

vector_store = FAISS.from_documents(
    embedding=OpenAIEmbeddings(),
    documents=docs
)

retriever1 = vector_store.as_retriever(
    search_type="mmr",
    seacrh_kwargs={"k": 3, "lambda_mult": 1} #It will behave as simple similarity search
)

retriever2 = vector_store.as_retriever(
    search_type="mmr",
    seacrh_kwargs={"k": 3, "lambda_mult": 0} #it will give diverse results
)

query = "What is Langchain?"

results1 = retriever1.invoke(query)
results2 = retriever2.invoke(query)

print("Results with MMR (lambda_mult=1):")
print(results1)

print("\nResults with MMR (lambda_mult=0):")
print(results2)
