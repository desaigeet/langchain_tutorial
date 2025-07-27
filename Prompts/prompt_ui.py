from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

st.header("Summarization Tool")
user_input_paper = st.selectbox("Enter your paper:",
                                 ["Attention is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"])
user_input_style = st.selectbox("Enter the style you want to summarize in:", 
                                ["Simple", "Math Heavy", "Code Heavy"])
user_input_length = st.selectbox(
    "Select the length of the summary:",
    ["Short", "Medium", "Long"]
)

template = PromptTemplate(
    template="""
You are an expert in summarizing academic papers. SUmmarize the paper {paper} in the style of {style} with a {length} summary. Include relevant quations if needed. Do not deviate from the style and length specified.
""",
input_variables=["paper", "style", "length"],
validate_template=True
)
prompt = template.invoke({
    "paper": user_input_paper,
    "style": user_input_style,
    "length": user_input_style
})
if st.button:
    result = model.invoke(prompt)
    st.write(f"Summary:{result.content}") 