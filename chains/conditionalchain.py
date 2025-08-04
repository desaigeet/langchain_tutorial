from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

class Review(BaseModel):
    sentiment: Literal["pos", "neg"] = Field(description='Return the overall sentiment of the review, e.g., positive, negative')

parser1 = PydanticOutputParser(pydantic_object=Review)

prompt1 = PromptTemplate(
    template = "Give the review of the following product: {product_description}. \n {format_instructions}",
    input_variables= ["product_description"],
    partial_variables =  {"format_instructions": parser1.get_format_instructions()}
)

# review = """
#             The SwiftCharge Pro is a sleek, black, wireless charging pad with a minimalist design, making it an attractive and unobtrusive addition to any workspace or bedside table. 
#             It boasts a compact size and a smooth, non-slip surface. The SwiftCharge Pro is designed for devices that support Qi wireless charging and promises rapid charging speeds
        #  """
review = """
         The SwiftCharge Pro is terrible. It does not charge my phone properly and the design is cheap. I expected better quality for the price I paid.
        """        
classifier_chain = prompt1 | model | parser1

sentiment = classifier_chain.invoke({"product_description": review })

print(f"Sentiment: {sentiment.sentiment}")

parser2 = StrOutputParser()
prompt2 = PromptTemplate(
    template= "Write an approprite resposne to the following positive review: {review}.",
    input_variables=["review"],
)

prompt3 =PromptTemplate(
    template = "Write a response to the following negative review: {review}.",
    input_variables=["review"]
)
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "pos", prompt2 | model | parser2),
    (lambda x: x.sentiment == 'neg', prompt3 | model | parser2),
    RunnableLambda(lambda x: "Coundn't find a review")
)

response = branch_chain.invoke(sentiment)
print(f"Response: {response}")
