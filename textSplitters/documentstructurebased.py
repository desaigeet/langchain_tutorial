from langchain.text_splitter import  RecursiveCharacterTextSplitter, Language
text = """
class Person:
    def __init__(self, name, age):
        self.name = name      # instance variable for name
        self.age = age        # instance variable for age

    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

# Creating an object (initializing the class)
person1 = Person("Alice", 30)

# Calling a method on the object
person1.greet()

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 200,
    chunk_overlap = 0
)

result = splitter.split_text(text)

print(len(result))
print(result[1])
