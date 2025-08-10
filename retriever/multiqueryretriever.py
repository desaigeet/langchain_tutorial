from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.retrievers import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

embedding_model = OpenAIEmbeddings()

vectore_store = FAISS.from_documents(
    embedding=embedding_model,
    documents=all_docs
)

similarity_retriever = vectore_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    llm = ChatOpenAI(model="gpt-3.5-turbo"),
    retriever = similarity_retriever,
)

query = "What are some ways to improve health and well-being?"

similarity_results = similarity_retriever.invoke(query)
multiqury_results = multiquery_retriever.invoke(query)

print("Results from similarity retriever:")
print(similarity_results)

print("\nResults from multi-query retriever:")
print(multiqury_results)
