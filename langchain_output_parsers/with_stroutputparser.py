from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(repo_id = "meta-llama/Llama-3.1-8B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

# 1st prompt -> detailed report
template1 =  PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on following text. /n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

res = chain.invoke({'topic': 'cricket'})

print(res)


