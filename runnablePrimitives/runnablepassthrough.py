from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatOpenAI(
    model = "gpt-3.5-turbo",
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Generate a joke on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Explain the joke {joke}.",
    input_variables=["joke"] 
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)
joke_exp_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, model, parser)
}
)

chain = RunnableSequence(joke_gen_chain, joke_exp_chain)

result = chain.invoke({"topic": "Narendra Modi"})

print("Joke:", result["joke"])
print("Joke Explanation:", result["explanation"])
