from langchain_community.retrievers import WikipediaRetriever

query = "What are the timelines of world wars?"

retriever = WikipediaRetriever(k = 2, language = "en")

docs = retriever.invoke(query)

print(docs[0].page_content)