from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

documents = ["Virat Kohli is king of cricket", 
            "Sachin Tendulkar is god of cricket", 
            "MSD is the best captian of white ball cricket",
            "Rohit sharma is the hitman",
            "Jasprit Bumrah is the best cricketer in the world" 
            ]

query = "tell me who is best captain in cricket"

emd = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs_emd = emd.embed_documents(documents)
query_emd = emd.embed_query(query)

sim_scores = cosine_similarity([query_emd], docs_emd)[0]
score = 0
idx = -1

for i in range(len(sim_scores)):
    if sim_scores[i] > score:
        score = sim_scores[i]
        idx = i

print(query)
print(documents[idx])
print("similarity score is:", score)


