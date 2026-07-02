from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(repo_id = "meta-llama/Llama-3.1-8B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(template="Give me 5 facts about {topic} \n {format_instruction}", input_variables=['topic'], partial_variables={'format_instruction': parser.get_format_instructions()})

chain = template | model | parser

res = chain.invoke({'topic': 'cricket'})

print(res)
print(type(res))

#Disadvantage of jsonoutputparser is that we can't enforce schema here if we want to do so we need to use structredoutputparser
