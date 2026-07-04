from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
The morning sky shimmered with soft shades of gold as the town slowly came to life. People wandered through quiet streets, exchanging friendly greetings and stopping at small cafés for coffee before work. 

In the central park, children laughed while chasing colorful kites, and a gentle breeze carried the scent of blooming flowers. 

Nearby, an old musician played a familiar melody that blended with the sounds of birds singing in the trees. As the day unfolded, conversations, discoveries, and small acts of kindness filled the hours, reminding everyone that even ordinary moments could become meaningful memories.
"""

#please see YT video to learn about how this splitter works first by breaking into para then sen then words and last into char
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)

res = splitter.split_text(text)

print(res)