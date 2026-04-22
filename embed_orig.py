from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import json

with open("chunks1.json", "r") as f:
    chunks = json.load(f)

docs = [Document(page_content=chunk["text"], metadata={"chunk_index": chunk["chunk_index"]}) for chunk in chunks]

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

persist_directory = "chroma_db"

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_directory
)

vectorstore.persist()

# You can load the vectorstore later with:
# vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)


#if you want to get the document part from a search

# while True:
#     user_input=input('Search:')
#     if (user_input.lower()=='exit'):
#         break
#     query = user_input
#     results = vectorstore.similarity_search(query, k=1)
#     for i, doc in enumerate(results):
#         print(f"--- Result {i+1} ---")
#         print(doc.page_content)
#         print()
