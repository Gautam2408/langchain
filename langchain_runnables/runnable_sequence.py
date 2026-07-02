from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

template1 = PromptTemplate(
    template="Write a joke on following \n {topic}",input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Explain the following joke \n {joke}",input_variables=["joke"]
)

chain = RunnableSequence(template1, model, parser, template2, model, parser)

res = chain.invoke({'topic': "Indian Government"})

print(res)