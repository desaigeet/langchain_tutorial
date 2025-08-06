from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "data",
    glob = "**/*.pdf",
    loader_cls=PyPDFLoader,
)

documents = loader.load()

print("Loaded Document Length: ", len(documents))
