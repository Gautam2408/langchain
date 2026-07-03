from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(docs)

print(len(docs))

print(type(docs))

print(docs[0])

print(type(docs[0]))

print(docs[0].metadata)

print(docs[0].page_content)


llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

template = PromptTemplate(template="Write a summary on following poem \n {poem}", input_variables=["poem"])

chain = template | model | parser

res = chain.invoke({'poem' : docs[0].page_content})

print(res)

#or using Runnablelambda in order to make doc loader as part of chain
# def doc_loader(text_file):
#     loader = TextLoader(text_file["name"], encoding=text_file["enc"])
#     docs = loader.load()
#     return docs[0].page_content


# llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

# model = ChatHuggingFace(llm = llm)

# parser = StrOutputParser()

# template = PromptTemplate(template="Write a summary on following poem \n {poem}", input_variables=["poem"])

# chain = RunnableLambda(doc_loader) | template | model | parser

# res = chain.invoke({'name': 'cricket.txt', 'enc':  'utf-8'})

# print(res)

