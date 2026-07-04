from langchain_community.document_loaders import PyPDFLoader

#loads the pdf page wise i.e. in docs array each field is a doc object of 1 page with its own meta_data as well as content 
#It works well when pdf mostly contains text it won't work well when pdf contains scanned images of text
loader = PyPDFLoader("dl-curriculum.pdf")

docs = loader.load()

print(docs)