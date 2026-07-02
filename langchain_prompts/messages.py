from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct', huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

# SystemMessage: these are the prompts that user writes at the start of new conversation just to give context or assigning role to the AI 
# Humanmessage: these are the prompts except the first prompt
# AIMessage: Response from AI
messages = [
    SystemMessage(content='You are helpful assistant'),
    HumanMessage(content='Tell me about langchain')
]

result = model.invoke(messages)
messages.append(AIMessage(result.content))

print(messages)
