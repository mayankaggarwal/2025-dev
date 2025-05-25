from modeller import get_response_text
from dal import store_embeddings
from filereader import load_documents
from text_splitter import split_documents


def run_rag(pdf_path, query):
    print("Loading and processing documents...")
    docs = load_documents(pdf_path)
    chunks = split_documents(docs)

    print("Storing embeddings using Ollama...")
    vectordb_retriever = store_embeddings(chunks)
    while True:
        query = input("Enter Query 'Type exit to quit':")
        if query == "exit":
            break
        result = get_response_text(vectordb_retriever, query)
        print("Answer:\n", result["result"])
        # print("\nSources:")
        # for doc in result["source_documents"]:
        #     print(f"- {doc.metadata['source']}")
        print("\n📚 Sources:")
        for doc in result["source_documents"]:
            print("-", doc.metadata.get("source", "[Unknown]"))
    print('Quitting')