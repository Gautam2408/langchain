from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os

#RunnableLambda is used to convert any python  function into runnable so that it can become a part of a chain

def word_count(text):
    return len(text.split())

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

template = PromptTemplate(
    template="Write a joke on following \n {topic}",input_variables=["topic"]
)

joke_gen_chain = RunnableSequence(template, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

chain = RunnableSequence(joke_gen_chain, parallel_chain)

res = chain.invoke({'topic': 'AI'})

print(res)