from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents:list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80
    )
    return text_splitter.split_documents(documents)