from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os


# -------------------------- Static MULTI-TURN CONVERSATION ------------------------

load_dotenv()

llm = HuggingFaceEndpoint(repo_id = "meta-llama/Llama-3.1-8B-Instruct",  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

#-----------------------------------------------------------------------------------------------
# now the problem with this below chatbot is that it can't give current response based upon previous chat it has so to do that first we need to store the chat history
# User: hi
# AI: How are you today? Is there something I can help you with or would you like to chat?
# User: tell me which is greater between 2 and 20
# AI: The number 20 is greater than 2.
# User: now can you multiply the greater number with 20
# AI: However, I don't see a number provided. Can you please provide the two numbers you'd like to compare? I'll be happy to multiply the greater number by 20 for you.
#-----------------------------------------------------------------------------------------------

# while True:
#     user_input = input("User: ")
#     if user_input == "exit":
#         break

#     res = model.invoke(user_input)
#     print("AI:", res.content)

# to overcome above problem we will store the chat history in a list
# chat_history = []
# while True:
#     user_input = input("User: ")
#     if user_input == "exit":
#         break

#     chat_history.append(user_input)
#     res = model.invoke(chat_history)
    
#     chat_history.append(res.content)
#     print("AI:", res.content)


# now there is one problem that in above chat history we also need to store what was asked by user and what response did AI gave right now we are just storing everything as a string in that list but as soon chat histroy gets bigger we need to store both response and prompt seperately so we need a dictonary for each prompt as well as response where key will be either User or AI and then value will be either prompt or response respectively now here langchain provides us with built in classes called as Message (System Message, Human Message, AI message)

chat_history = [
    SystemMessage(content='You are a helpful assistant')
]
while True:
    user_input = input("User: ")
    if user_input == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))
    res = model.invoke(chat_history)
    
    chat_history.append(AIMessage(res.content))
    print("AI:", res.content)

print(chat_history)