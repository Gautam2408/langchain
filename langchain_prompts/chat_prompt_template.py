from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# -------------------------- Dynamic MULTI-TURN CONVERSATION ------------------------
# When we want to create a dynamic HumanMessage, SystemMessage 

# Below code won't work
# chat_template = ChatPromptTemplate([
#     SystemMessage(content='You are a helpful {domain} expert'),
#     HumanMessage(content='Explain in simple terms, what is {topic}')
# ])

# prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'doosra'})

# print(prompt)

# Correct code
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'doosra'})

print(prompt)