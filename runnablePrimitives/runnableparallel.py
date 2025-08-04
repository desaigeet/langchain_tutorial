from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

model = ChatOpenAI(
    model = "gpt-3.5-turbo",
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= "Generate a tweet about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template= "Generate a linkedin post about {topic}",
    input_variables=["topic"]
)

chain = RunnableParallel(
    {
    "tweet" : RunnableSequence(prompt1, model, parser),
    "linkedin" : RunnableSequence(prompt2, model, parser)
    }
)
result = chain.invoke({"topic": "Narendra Modi"})

print("Tweet:", result["tweet"])
print("Linkledin Post:", result["linkedin"])
