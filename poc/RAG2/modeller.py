from langchain.chains import RetrievalQA
from langchain.llms.ollama import Ollama

def get_response_text(retriever, query):
    qa_chain = RetrievalQA.from_chain_type(llm=get_llm_model(), retriever=retriever, return_source_documents=True)
    return qa_chain({"query": query})

def get_llm_model():
    return Ollama(model="llama3.2") 