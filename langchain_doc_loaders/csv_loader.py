from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="Social_Network_Ads.csv")

#each row will have a doc class so the len of docs array depends upon no of rows in csv file
docs = loader.load()

print(docs)