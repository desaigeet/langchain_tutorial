from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model = "gpt-3.5-turbo",
)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= "Generate ablog about the topic: {topic}",
    input_variables= ["topic"]
)

prompt2 = PromptTemplate(
    template= "Generate a summary of the blog: {blog}",
    input_variables= ["blog"]
)
def word_count(text):
    return len(text.split())

blog_gen_chain = RunnableSequence(prompt1, model, parser)

blog_summary_chain = RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(blog_gen_chain, blog_summary_chain)
result = final_chain.invoke({"topic": "Narendra Modi"})
print("Blog:", result)
