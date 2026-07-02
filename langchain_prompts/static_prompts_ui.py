from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(repo_id = "meta-llama/Llama-3.1-8B-Instruct", huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

st.header("Research Tool")

usr_input = st.text_input("Enter Your prompt")

if st.button("summarize"):
    res = model.invoke(usr_input)
    st.write(res.content)


