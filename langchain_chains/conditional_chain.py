from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()
llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")

parser1 = PydanticOutputParser(pydantic_object=Feedback)

template1 = PromptTemplate(template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}", input_variables=["feedback"], partial_variables={'format_instruction': parser1.get_format_instructions()})

parser2 = StrOutputParser()

classifier_chain = template1 | model | parser1

template2 = PromptTemplate(template="Write an appropriate response to this below positive feedback \n {feedback}", input_variables=["feedback"])

template3 = PromptTemplate(template="Write an appropriate response to this below negative feedback \n {feedback}", input_variables=["feedback"])


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", template2 | model | parser2),
    (lambda x: x.sentiment == "negative", template3 | model | parser2),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

res = chain.invoke({'feedback': "this is a terrible phone"})

print(res)


