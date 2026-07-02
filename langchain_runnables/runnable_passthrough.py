from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
import os

#RunnablePassthrough is used to give input as exact output
#scenario: humne runnable_sequence mai llm se ek topic par joke create karwaya aur phir use explain bhi karwaya but last output jo thha woh bass joke ko explain karneko thha so what if abb mujhe joke and uske explanation ko parallely dikhana ho
 

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

joke_gen_chain = RunnableSequence(template1, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explaination': RunnableSequence(template2, model, parser)
})

chain = RunnableSequence(joke_gen_chain, parallel_chain)

res = chain.invoke({"topic": "AI"})

print(res)