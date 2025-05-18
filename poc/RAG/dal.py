from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from custom_embedding_function import get_embedding_function

#from custom_embedding_function import get_embedding_function

def add_to_chroma(chunks: list[Document]):
    db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        collection_name="monopoly-rag"
    )

def get_matching_results(query_text:str):
    db = Chroma(
        embedding_function=get_embedding_function(),
        collection_name="monopoly-rag"
    )
    return db.similarity_search_with_score(query_text, k=5)

