import argparse
from modeller import get_response_text
from dal import add_to_chroma, get_matching_results
from text_splitter import split_documents
from filereader import load_documents


def Run():
    print("In RAG")
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    #query = getQuery()
    while True:
        query = input("Enter Query 'Type exit to quit':")
        if query == "exit":
            break
        results = get_matching_results(query)
        response_text = get_response_text(query,results)
        print(response_text)
    print('Quitting')

def getQuery():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    return args.query_text
