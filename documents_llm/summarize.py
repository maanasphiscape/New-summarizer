from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def summarize_document(docs: list[Document], model_name: str, openai_api_key: str,
                       base_url: str, temperature: float = 0.1) -> str:
    llm = ChatOpenAI(temperature=temperature, model_name=model_name, api_key=openai_api_key, base_url=base_url)
    prompt = PromptTemplate.from_template(
        "Write a long summary of the following document.\nOnly include information that is part of the document. "
        "Do not include your own opinion or analysis.\n\nDocument:\n\"{document}\"\nSummary:"
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="document")
    result = stuff_chain.invoke(docs)
    return result["output_text"]
