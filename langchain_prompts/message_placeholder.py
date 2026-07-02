from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, messages_to_dict,messages_from_dict
import json

# showing how previous chat would have been stored into a json file 
# array of objects
histroy = [
    HumanMessage(content="I want to request a refund for my order #12345."),
    AIMessage(content="Your refund request for order #12345 has been initiated. It will be processed in 3-5 business days.")
]

#write
with open("chat_history.json", "w") as f:
    json.dump(messages_to_dict(histroy), f)



#load chat history
chat_history = []
with open('chat_history.json') as f:
    chat_history = messages_from_dict(json.load(f))

#chat template
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

#create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund'})

print(prompt)


#Message Placeholder: is a placeholder jaha par hum set of messages ke liye ek placeholder creae karte hai generally hum isko use karte hai to retrieve and store chat histroy