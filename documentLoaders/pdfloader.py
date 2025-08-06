from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/lekl101.pdf")

documents = loader.load()

print(f"Page count: {len(documents)}")
print("First document content: ", documents[0].page_content)
