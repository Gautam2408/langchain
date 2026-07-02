from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()

llm1 = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model1 = ChatHuggingFace(llm = llm1)

llm2 = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model2 = ChatHuggingFace(llm = llm2)

parser = StrOutputParser()

template1 = PromptTemplate(
    template="Create a tweet on following topic \n {topic}",input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Create a linkedin post following topic \n {topic}",input_variables=["topic"]
)

chain = RunnableParallel({
    'tweet': RunnableSequence(template1, model1, parser),
    'linkedin': RunnableSequence(template2, model2, parser),
})

res = chain.invoke({'topic': "Indian Government"})

print(res)