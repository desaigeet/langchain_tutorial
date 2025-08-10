from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()


doc1 = Document(
    page_content="""
    Sachin Tendulkar, known as the "Master Blaster," is a cricketing legend with over 100 international centuries.
    He inspired generations with his discipline, humility, and unmatched technique.
    For many fans, he is the heartbeat of Indian cricket.
    """,
    metadata={"team": "India", "player": "Sachin Tendulkar", "role": "Batsman"}
)

doc2 = Document(
    page_content = """
    Virat Kohli, a modern-day batting maestro, has redefined aggression in cricket.
    With over 70 international centuries, he is a symbol of excellence and passion.
    His leadership and consistency make him a pivotal figure in Indian cricket.
    """,
    metadata={"team": "India", "player": "Virat Kohli", "role": "Batsman"}
)

doc3 = Document(
    page_content="""
   Jasprit Bumrah, India's premier fast bowler, is known for his unique bowling action and death-over prowess.
   With a knack for taking crucial wickets, he has become a key player in all formats.
   His ability to bowl yorkers at will makes him a nightmare.
   """,
    metadata={"team": "India", "player": "Jasprit Bumrah", "role": "Bowler"}
)

doc4 = Document(
    page_content = """
   Bret Lee, the Australian fast bowler, was known for his express pace and aggressive bowling style.
   With a career spanning over a decade, he was a key player in Australia's dominance in world cricket.
   His fiery spells and wicket-taking ability made him a fan favorite.
         """,
    metadata={"team": "Australia", "player": "Bret Lee", "role": "Bowler"}
)

embedding_model = OpenAIEmbeddings()

vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="vectorStores/chroma_db",
    collection_name="cricket_players",
)

vector_store.add_documents([doc1, doc2, doc3, doc4])

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query = "Who is India's best bowler?"

results = retriever.invoke(query)

print(len(results))

print("Results:")

for i, doc in enumerate(results):
    print(f"Document {i+1}:")
    print("Content:", doc.page_content)
