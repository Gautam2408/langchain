from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(repo_id = "meta-llama/Llama-3.1-8B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name='fact#1', description='fact 1 about the topic'),
    ResponseSchema(name='fact#2', description='fact 2 about the topic'),
    ResponseSchema(name='fact#3', description='fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(template="Give me 3 facts about {topic} \n {format_instruction}", input_variables=['topic'], partial_variables={'format_instruction': parser.get_format_instructions()})

chain = template | model | parser

res = chain.invoke({'topic': 'cricket'})

print(res)
print(type(res))

#Disadvantage of structuredoutputparser is that it doesn't enforce data validation although it enforces schema so for data validation we need to use pydenticoutputparseer
