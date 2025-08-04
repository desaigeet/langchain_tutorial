from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model = "gpt-3.5-turbo",
)

prompt = PromptTemplate(
    template = "Write a joke about the topic: {topic}",
    input_variables= ["topic"]
)

parser = StrOutputParser()

def word_count(text):
    return len(text.split())

joke_gen_chain = RunnableSequence(prompt, model, parser)
joke_count_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(word_count)
    }
)

final_chain = RunnableSequence(joke_gen_chain, joke_count_chain)
result = final_chain.invoke({"topic": "Narendra Modi"})

print("Joke:", result["joke"])
print("Word Count:", result["word_count"])
