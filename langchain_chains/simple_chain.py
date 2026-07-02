from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task="text-generation")

model = ChatHuggingFace(llm = llm)

template = PromptTemplate(template="Give me 5 interesting facts about {topic}", input_variables=["topic"])

parser = StrOutputParser()

chain = template | model | parser

res = chain.invoke({'topic': 'cricket'})

print(type(res))

# chain.get_graph().print_ascii()