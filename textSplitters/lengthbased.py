from langchain.text_splitter import  CharacterTextSplitter

text = """
This is a long text that we will use to test the length-based text splitter.
It contains multiple sentences and paragraphs to ensure that the text splitter can handle various lengths of text effectively.
The text splitter should be able to split this text into smaller chunks based on the specified length.
The goal is to create manageable pieces of text that can be processed individually.
This is another paragraph to add more content to the text.
"""

splitter = CharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 0,
    separator=''
)

result = splitter.split_text(text)

print(result)
