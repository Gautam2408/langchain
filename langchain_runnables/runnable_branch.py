from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

template1 = PromptTemplate(template="Write a detailed report on {topic}", input_variables=["topic"])

template2 = PromptTemplate(template="Generate a summary on the following text \n {text}", input_variables=["text"])

report_gen_chain = RunnableSequence(template1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(template2, model, parser)),
    RunnablePassthrough()
    )

chain = RunnableSequence(report_gen_chain, branch_chain)

res = chain.invoke({"topic": "Fifa Worldcup 2026"})

print(res)

chain.get_graph().print_ascii()