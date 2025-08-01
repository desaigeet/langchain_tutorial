from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a {domain} expert.'),
    ('human', 'Explain {topic} in simple words.'),
    ]
)

prompt = chat_template.invoke({
    'domain': 'Python',
    'topic': 'decorators'
})

print(prompt)
