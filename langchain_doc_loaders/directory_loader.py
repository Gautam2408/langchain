from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path="book", glob="*.pdf", loader_cls=PyPDFLoader)

#see the difference b/w loader and lazy_loader by running following code two codes
#code-1
# docs = loader.load()

#code-2
docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)