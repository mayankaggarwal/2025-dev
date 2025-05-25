from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from custom_embedding_function import get_embedding_function

def store_embeddings(docs: list[Document],persist_directory = "db"):
    vectordb = Chroma.from_documents(
        documents=docs, 
        embedding=get_embedding_function(),
        persist_directory=persist_directory)
    vectordb.persist()
    return vectordb.as_retriever(search_kwargs={"k": 3})