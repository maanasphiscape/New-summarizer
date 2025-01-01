from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def summarize_document(docs, model_name, base_url, temperature=0.1):
    prompt_template = """
        Write a long summary of the document below:
        "{document}"
        Summary:
    """
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        base_url=base_url,
    )
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="document")
    result = stuff_chain.invoke(docs)
    return result["output_text"]
