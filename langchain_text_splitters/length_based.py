from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# text = """
# The morning sky shimmered with soft shades of gold as the town slowly came to life. People wandered through quiet streets, exchanging friendly greetings and stopping at small cafés for coffee before work. In the central park, children laughed while chasing colorful kites, and a gentle breeze carried the scent of blooming flowers. Nearby, an old musician played a familiar melody that blended with the sounds of birds singing in the trees. As the day unfolded, conversations, discoveries, and small acts of kindness filled the hours, reminding everyone that even ordinary moments could become meaningful memories.
# """

# splitter = CharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=0,
#     separator=""
# )

# res = splitter.split_text(text=text)

# print(res)

loader = PyPDFLoader(file_path="dl-curriculum.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)

#list of document objects
res = splitter.split_documents(docs)

print(res[0].page_content)