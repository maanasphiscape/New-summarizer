from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def query_document(
    docs: list[Document],
    user_query: str,
    model_name: str,
    openai_api_key: str,
    base_url: str,
    temperature: float = 0.3,
) -> str:
    llm = ChatOpenAI(
        temperature=temperature,
        model_name=model_name,
        api_key=openai_api_key,
        base_url=base_url,
    )
    chain = get_map_reduce_chain(llm, user_query=user_query)
    result = chain.invoke(docs)
    return result["output_text"]


def get_map_reduce_chain(llm: ChatOpenAI, user_query: str) -> MapReduceDocumentsChain:
    map_template = """The following is a set of documents
    {docs}
    Based on this list of documents, please identify the information that is most relevant to the following query:
    {user_query} 
    If the document is not relevant, please write "not relevant".
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template).partial(user_query=user_query)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    reduce_template = """The following is a set of partial answers to a user query:
    {docs}
    Take these and distill it into a final, consolidated answer to the following query:
    {user_query} 
    Complete Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template).partial(user_query=user_query)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
    )
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )
