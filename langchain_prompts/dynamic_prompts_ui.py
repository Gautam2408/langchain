from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(repo_id = "meta-llama/Llama-3.1-8B-Instruct", huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task = "text-generation")

model = ChatHuggingFace(llm = llm)

st.header("Research Tool")

ppr_input = st.selectbox("Selec Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

sty_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

len_input = st.selectbox("Select Explanation Lenght", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanations)"])

# one way
# template  = PromptTemplate(
#     template='''
#     Please summarize the research paper titled "{paper_input}" with the following specifications:
#     Explanation Style: {style_input}  
#     Explanation Length: {length_input}  
#     1. Mathematical Details:  
#     - Include relevant mathematical equations if present in the paper.  
#     - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
#     2. Analogies:  
#     - Use relatable analogies to simplify complex ideas.  
#     If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
#     Ensure the summary is clear, accurate, and aligned with the provided style and length.
#     ''',
#     input_variables=['paper_input', 'style_input', 'length_input'],
#     # ye make sure karega ki humne template mai jitne variables diye hai utne variables hum input_variables mai mention kar rhe hai ya nhi
#     validate_template=True
# )


#second way: if we want to same template to be reused in  multiple files then we will convert it into json and load that json wherever this same template is required 

template = load_prompt('template.json')

# prompt = template.invoke({
#     'paper_input': ppr_input,
#     'style_input': sty_input,
#     'length_input': len_input
# })

# if st.button("summarize"):
#     res = model.invoke(prompt)
#     st.write(res.content)


# upar humne do baar invoke call kiya hai now with the help of chain (jisko ki PromptTemplate support karta hai) hum ek baar hi invoke use karenge 

if st.button("summarize"):
    chain = template | model
    res = chain.invoke({
        'paper_input': ppr_input,
        'style_input': sty_input,
        'length_input': len_input
    })

    st.write(res.content)