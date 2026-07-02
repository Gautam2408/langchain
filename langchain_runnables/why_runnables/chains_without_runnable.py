import random

class NakliLLM:

  def __init__(self):
    print('LLM created')

  def predict(self, prompt):

    response_list = [
        'Delhi is the capital of India',
        'IPL is a cricket league',
        'AI stands for Artificial Intelligence'
    ]

    return {'response': random.choice(response_list)}
  

class NakliPromptTemplate:

  def __init__(self, template, input_variables):
    self.template = template
    self.input_variables = input_variables

  def format(self, input_dict):
    return self.template.format(**input_dict)
  

#manually connecting two different components as connecting these two components for building an application was too common for AI engineers and to make thier life easier langchain developers built many chain classes one per repetitive use case(like below NakliLLMChain)
# template = NakliPromptTemplate(
#     template='Write a {length} poem about {topic}',
#     input_variables=['length', 'topic']
# )

# prompt = template.format({'length':'short','topic':'india'})

# llm = NakliLLM()

# res = llm.predict(prompt)

# print(res)


class NakliLLMChain:

  def __init__(self, llm, prompt):
    self.llm = llm
    self.prompt = prompt

  def run(self, input_dict):

    final_prompt = self.prompt.format(input_dict)
    result = self.llm.predict(final_prompt)

    return result['response']
  

template = NakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

llm = NakliLLM()

chain = NakliLLMChain(llm, template)

res = chain.run({'length':'short', 'topic': 'india'})

print(res)

#but the problem is that as these components were not standardised(having different function name like here NakliLLM uses predict function; NakliPromptTemplate uses format function; NakliLLMChain uses run function) due to which these chain classes were not flexible(for ex: use llm twice one for creating a joke on topic and then for explaining that joke, so how can we pass same two llm object in that run function so to tackle it runnable classes came into the  scenario)
