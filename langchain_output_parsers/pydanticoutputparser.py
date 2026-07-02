from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: Optional[str] = Field(default = None, description="Name of the city in which person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(template="Generate Name, Age and city of a fictional {nationality} person \n {format_instruction}", input_variables=['nationality'], partial_variables={'format_instruction':parser.get_format_instructions()})

chain = template | model | parser

res = chain.invoke({'nationality' : 'amercian'})

print(res)