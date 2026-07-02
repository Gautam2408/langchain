from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

doc = ["Delhi is the capital of India", "MSD is my hero", "Messi is goat"]

res = embedding.embed_documents(doc)

print(str(res))