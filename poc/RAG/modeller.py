from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_response_text(query_text:str,matching_results:list[Document]):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in matching_results])
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt)
    return response_text
