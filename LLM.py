from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama 
import json
persist_directory = "chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

llm = ChatOllama(model="llama3.1:latest")  

# Linking vectorstore & LLM using RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True
)

while True:
    user_input = input("query: ")
    if user_input.lower() == "exit":
        break

    response = qa_chain(user_input)
    print("\n Answer:", response["result"])
    print("Source documents:\n")
    for i, doc in enumerate(response["source_documents"]):
        print(f"-Source {i+1} -")
        print(doc.page_content)
        print()

