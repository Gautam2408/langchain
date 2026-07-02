from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm1 = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task="text-generation")
model1 = ChatHuggingFace(llm = llm1)

llm2 = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct", huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task="text-generation")
model2 = ChatHuggingFace(llm = llm2)

model3 = ChatGroq(model="openai/gpt-oss-120b")

template1 = PromptTemplate(template="Generate short and simple notes from following text \n {topic}", input_variables=['topic'])

template2 = PromptTemplate(template="Generate 5 short question answers from following text \n {topic}", input_variables=['topic'])

template3 = PromptTemplate(template="Merge the below provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}", input_variables=['notes', 'quiz'])

