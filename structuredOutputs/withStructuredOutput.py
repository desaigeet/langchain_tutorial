from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

class Review_typedict(TypedDict):
    key_theme: Annotated[str, 'The main theme or topic of the review']
    summary: Annotated[str,'A brief summary of the product review']
    sentiment: Annotated[str,'The overall sentiment of the review, e.g., positive, negative, neutral']

structured_model = model.with_structured_output(Review_typedict)

result = structured_model.invoke("""
                    The SwiftCharge Pro is a sleek, black, wireless charging pad with a minimalist design, making it an attractive and unobtrusive addition to any workspace or bedside table. 
                    It boasts a compact size and a smooth, non-slip surface. The SwiftCharge Pro is designed for devices that support Qi wireless charging and promises rapid charging speeds
                    """)

print(result)
