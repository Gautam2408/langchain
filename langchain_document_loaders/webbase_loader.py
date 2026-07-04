from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

url = ["https://zolostays.com/pgs-in-bannerghatta-bangalore", "https://www.flipkart.com/acer-aspire-3-intel-core-i3-13th-gen-1305u-8-gb-512-gb-ssd-windows-11-home-a324-53-thin-light-laptop/p/itma6a6812aa712e?pid=COMH4B6CGTPNVJW9&lid=LSTCOMH4B6CGTPNVJW9JOF4ZC&marketplace=FLIPKART&store=6bo%2Fb5g&srno=b_1_1&otracker=browse&fm=neo%2Fmerchandising&iid=2e88e6bb-c9fe-4ffb-a348-b30153a8ce4c.COMH4B6CGTPNVJW9.SEARCH&ppt=clp&ppn=new-elec-clp-march-at-store&ssid=zl08x128q80000001783082555956&ov_redirect=true"]

loader = WebBaseLoader(url)

#docs will contain doc class for each url, so here len of docs array is 2 beacuse we have 2 urls
docs = loader.load()

# print(docs)

# print(len(docs))

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

template = PromptTemplate(template="Answer the question {question} based upon following text \n {text}", input_variables=["question", "text"])

chain = template | model | parser

res1 = chain.invoke({'question': "How many properties it is showing near Bannerghatta", 'text': docs[0].page_content})

print(res1)

res2 = chain.invoke({'question': "What is this page about", 'text': docs[1].page_content})

print(res2)