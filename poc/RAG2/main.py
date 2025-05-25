from rag import run_rag


def main():
    run_rag("documents/dotnet_aspire.pdf", "What is the main idea of this document?")

if __name__ == "__main__":
    main()