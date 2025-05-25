from langchain.document_loaders import PyPDFLoader
import pdfplumber
import spacy
from langchain.docstore.document import Document
nlp = spacy.load("en_core_web_sm")

# def load_documents(file_path):
#     loader = PyPDFLoader(file_path)
#     return loader.load()

def load_documents(file_path):
    text, tables = extract_text_from_pdf(file_path)

    # Optionally append table content to text (flattened)
    for i, table in enumerate(tables):
        table_str = "\n".join([", ".join(row) for row in table if row])
        text += f"\n\n[Table {i+1}]\n{table_str}"

    return [Document(page_content=text)]

def extract_text_from_pdf(pdf_path):
    """Extracts text, detects tables, and improves sentence-level coherence."""
    full_text = []
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract main text
            page_text = page.extract_text(x_tolerance=1.5)
            if page_text:
                full_text.append(page_text)

            # Extract tables
            page_tables = page.extract_tables()
            if page_tables:
                tables.extend(page_tables)

    raw_text = "\n\n".join(full_text)

    # Sentence-level cleanup using SpaCy
    doc = nlp(raw_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    return "\n".join(sentences), tables