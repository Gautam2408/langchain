from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm1 = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task="text-generation")
model1 = ChatHuggingFace(llm = llm1)

llm2 = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct", huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), task="text-generation")
model2 = ChatHuggingFace(llm = llm2)

model3 = ChatGroq(model="openai/gpt-oss-120b")

template1 = PromptTemplate(template="Generate short and simple notes from following text \n {topic}", input_variables=['topic'])

template2 = PromptTemplate(template="Generate 5 short question answers from following text \n {topic}", input_variables=['topic'])

template3 = PromptTemplate(template="Merge the below provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}", input_variables=['notes', 'quiz'])

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': template1 | model1 | parser,
    'quiz': template2 | model2 | parser
})

sequential_chain = template3 | model3 | parser

merged_chain = parallel_chain | sequential_chain

topic = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

res = merged_chain.invoke({
    'topic': topic
})

print(res)

merged_chain.get_graph().print_ascii()